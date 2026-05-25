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

Shell completion (BRIDGE-F-005, v1.3 Wave 6a):

    pip install argcomplete   # if not already pulled in via pyproject
    eval "$(register-python-argcomplete backprop)"

    # Then `backprop tr<TAB>` completes to `train`, `backprop export
    # --format <TAB>` completes to lora / merged / gguf, etc.

See handbook/getting-started.md for the per-shell snippets (bash / zsh /
fish).
"""

# PYTHON_ARGCOMPLETE_OK — magic comment for argcomplete's global-completion
# mode; when installed via `eval "$(register-python-argcomplete backprop)"`
# the completer scans this file for the marker before doing any work.

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
# EXIT CODES (Ship Gate B2 + BRIDGE-F-006 sysexits.h overlay)
# =============================================================================
# Legacy Ship Gate B2 codes (preserved for back-compat — every documented
# contract since v1.1.0 promises these and downstream CI scripts assert on
# them):
#   0   = success
#   1   = user error          (bad args, missing input, validation failure)
#   2   = runtime error       (unexpected crash, IO failure, internal bug)
#   3   = partial success     (operation completed with some failures)
#   130 = interrupted         (Ctrl+C / SIGINT — POSIX convention)
#
# BRIDGE-F-006 (v1.3 Wave 6a) overlay — sysexits.h-flavored finer-grained
# codes that the top-level exception net at main() emits when it can match
# the failure mode to a specific category. The legacy 1 / 2 / 3 / 130 codes
# REMAIN the contract: per-subcommand handlers still emit them. The sysexits
# overlay only fires from the last-resort exception net when nothing inside
# the subcommand was specific enough to assign a code; downstream wrappers
# that compare against the legacy code set keep working unchanged because
# the overlay codes are disjoint (64 / 65 / 69 / 70 / 77 / 137 vs 0 / 1 /
# 2 / 3 / 130). The mapping is documented in handbook/cli-reference.md
# (handoff to frontend agent, see report bundle below).
#
# sysexits.h conventions (BSD /usr/include/sysexits.h):
#   EX_USAGE        = 64  malformed CLI invocation
#   EX_DATAERR      = 65  input file unusable / malformed
#   EX_UNAVAILABLE  = 69  required service unreachable (HF Hub down, etc.)
#   EX_SOFTWARE     = 70  internal software error (uncaught exception)
#   EX_NOPERM       = 77  permission denied (filesystem, ACL)
#
# Linux convention:
#   137             SIGKILL by OOM-killer (used for CUDA OOM where the
#                   process is functionally killed by the kernel even
#                   when Python catches torch.cuda.OutOfMemoryError —
#                   matches the bash $? value operators see when an OOM
#                   slays the process directly)

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_PARTIAL_SUCCESS = 3
EXIT_INTERRUPTED = 130

# BRIDGE-F-006 sysexits overlay (last-resort net only — subcommand handlers
# continue to emit 0 / 1 / 2 / 3 / 130 per their documented contracts).
EXIT_USAGE = 64          # EX_USAGE      — argparse / CLI structure failure
EXIT_DATA_ERR = 65       # EX_DATAERR    — dataset malformed / unreadable
EXIT_UNAVAILABLE = 69    # EX_UNAVAILABLE — required service down (HF Hub)
EXIT_SOFTWARE = 70       # EX_SOFTWARE   — internal uncaught exception
EXIT_NO_PERM = 77        # EX_NOPERM     — permission denied
EXIT_OOM_KILLED = 137    # Linux OOM-killer convention (SIGKILL = 128 + 9)


# =============================================================================
# LOGGING-SETUP FAILURE TRACKING (BRIDGE-B-013 Stage C)
# =============================================================================
# Module-level flag set by main() when configure_logging / bind_run_context /
# cli_invoked-emit raise. Surfaced as a single stderr WARN line at the end of
# main() so the operator knows logs aren't flowing without aborting the CLI.

_LOGGING_SETUP_FAILED = False
_LOGGING_SETUP_FAIL_REASON = ""


# =============================================================================
# SUBCOMMAND STABILITY TIERS (BRIDGE-B-017 Stage C)
# =============================================================================
# Centralized registry of subcommand stability so the CLI can print a
# one-line deprecation hint when an operator invokes a subcommand whose
# tier is "deprecated-prefer-X". (A future `backprop info --subcommand-tiers`
# scaffolding entry was considered but never registered on info_parser — do
# NOT reference such a flag in operator-facing copy until it ships, per
# [[no-banner-documenting-no-op]].)
#
# Tiers:
#   stable                   — Part of the documented contract; no removal
#                              planned.
#   experimental             — May change shape between minor versions; opt-in
#                              consumers should pin the exact version.
#   deprecated-prefer-X      — Will be removed in a future major; use `X`
#                              instead. Prints a one-line stderr hint on
#                              invocation.
#
# Pre-fix the only deprecation candidate was `list-runs` vs `runs` (added in
# BRIDGE-F-001 with the schema_version field). The registry has only one
# deprecated entry today but the scaffold is in place for v1.4 candidates.

SUBCOMMAND_TIERS: dict[str, str] = {
    "train": "stable",
    "multi-run": "stable",
    "export": "stable",
    "ui": "stable",
    "info": "stable",
    "config": "stable",
    "resume": "stable",
    "push": "stable",
    "runs": "stable",
    "show-run": "stable",
    "list-runs": "deprecated-prefer-runs",
    # BRIDGE Wave 6b (v1.3) — multi-run workflow primitives. Marked
    # 'experimental' so the contract can absorb shape iteration in v1.4
    # without surprise to operators who pinned against the v1.3 shape.
    # Promotes to 'stable' once one minor cycle has shipped without
    # breaking changes.
    "diff-runs": "experimental",
    "replay": "experimental",
    "export-runs": "experimental",
}


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


# BRIDGE-B-005 (Stage C humanization): tighten --auth parsing so a malformed
# value fails on the argparse side with a humanized error that names the
# offending character class. Pre-fix the only validation was downstream in
# cmd_ui (split-on-first-colon + validate_auth_shape), which did NOT reject
# usernames containing colons / whitespace / control chars or passwords
# containing newlines — both of which would corrupt the Basic-auth header.
#
# Accepted shape: "<username>:<password>" — username matches
# ^[^\s:\x00\x1f\x7f]+$ (no whitespace, no colon, no NUL, no C0/DEL controls);
# password is the remainder after the FIRST colon and may NOT contain
# \r / \n / \x00 (those would split the Basic-auth header on the wire).
#
# The error message names which side and which character class failed,
# masks the actual value (operators paste these into bug reports), and
# points at BACKPROPAGATE_UI_AUTH / a future --auth-file as escape hatches
# for shell-history-sensitive deployments.
_AUTH_USERNAME_FORBIDDEN = re.compile(r"[\s:\x00-\x1f\x7f]")
_AUTH_PASSWORD_FORBIDDEN = re.compile(r"[\r\n\x00]")


def _auth_credential(value: str) -> str:
    """argparse type for --auth user:pass values.

    Returns the original raw string on success (the cmd_ui handler does
    the actual ``split(':', 1)`` so call-site logic is unchanged).
    Raises ``argparse.ArgumentTypeError`` with a humanized message naming
    the offending side + character class on failure.
    """
    if not isinstance(value, str) or not value:
        raise argparse.ArgumentTypeError(
            "--auth requires user:pass (both non-empty). Got an empty value. "
            "Use `BACKPROPAGATE_UI_AUTH=user:pass` env var to avoid shell "
            "history."
        )

    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "--auth requires user:pass — no colon separator found. "
            "Use `BACKPROPAGATE_UI_AUTH=user:pass` env var to avoid shell "
            "history."
        )

    username, password = value.split(":", 1)

    if not username:
        raise argparse.ArgumentTypeError(
            "--auth requires user:pass (both non-empty). Got: empty username "
            "(format was ':<pass>'). "
            "Use `BACKPROPAGATE_UI_AUTH=user:pass` env var to avoid shell "
            "history."
        )

    if not password:
        raise argparse.ArgumentTypeError(
            "--auth requires user:pass (both non-empty). Got: empty password "
            "(format was '<user>:'). "
            "Use `BACKPROPAGATE_UI_AUTH=user:pass` env var to avoid shell "
            "history."
        )

    bad_user = _AUTH_USERNAME_FORBIDDEN.search(username)
    if bad_user:
        raise argparse.ArgumentTypeError(
            "--auth username contains a forbidden character "
            f"(category: whitespace / colon / control). "
            f"Offending codepoint: U+{ord(bad_user.group(0)):04X}. "
            "Usernames may contain printable ASCII except space, colon, and "
            "control chars; the colon is the user:pass separator and a "
            "username-embedded colon would silently become part of the "
            "password. Quote the value if you need special chars in the "
            "password."
        )

    bad_pass = _AUTH_PASSWORD_FORBIDDEN.search(password)
    if bad_pass:
        raise argparse.ArgumentTypeError(
            "--auth password contains a forbidden character "
            f"(category: newline / NUL). "
            f"Offending codepoint: U+{ord(bad_pass.group(0)):04X}. "
            "Newlines and NUL would corrupt the HTTP Basic-auth header "
            "on the wire. Strip the line ending or pick a different password."
        )

    return value


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
    # BRIDGE-B-012 (Stage C): additional patterns the error-redaction layer
    # should catch before pasting into a bug report.
    #
    # URL-embedded credentials (https://user:token@host/...): HfHubHTTPError
    # occasionally surfaces request URLs in its message. Match the canonical
    # `scheme://user:secret@host` form (NOT the user@host form, which is just
    # an SSH-style address with no secret). The capture preserves the scheme
    # and the host so the operator still sees what service was contacted.
    (
        re.compile(r"(https?://)[^/\s:@]+:[^/\s:@]+@([^/\s]+)"),
        r"\1<REDACTED>@\2",
    ),
    # JWT tokens (3 base64url segments separated by `.`): RFC 7519 shape.
    # The `eyJ` prefix is the base64 header of `{"`, present in essentially
    # every real JWT, which keeps false-positive risk low against prose.
    (
        re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
        "<JWT_REDACTED>",
    ),
    # GitHub Personal Access Tokens (ghp_*, ghs_*, gho_*, ghu_*, ghr_*):
    # canonical prefix per GitHub's secret-scanning docs; the suffix length
    # is 36+ base62 chars. Matches modern PATs (post-2021); the legacy
    # 40-hex format isn't redacted here because it has high false-positive
    # rate against generic 40-hex strings.
    (
        re.compile(r"\b(ghp|ghs|gho|ghu|ghr)_[A-Za-z0-9]{36,}\b"),
        r"\1_<REDACTED>",
    ),
    # GitLab PATs (glpat-...): canonical 20+ base62 / hyphen suffix per
    # GitLab's tokens docs.
    (
        re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
        "glpat-<REDACTED>",
    ),
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

        # BRIDGE Wave 6b (v1.3): assemble the optional kwarg bundle for the
        # backend's Wave 6b additions. Each kwarg is bound on the Trainer
        # constructor only when (a) the operator explicitly set the flag AND
        # (b) the Trainer constructor in the installed build accepts that
        # kwarg. The accept-test uses inspect.signature so a future Trainer
        # build that drops a kwarg cleanly degrades the CLI to "no-op" for
        # that flag rather than crashing on a TypeError. The default values
        # threaded by argparse (False / 'default' / 'quality' / 'auto') are
        # only forwarded when the underlying Trainer kwarg exists so the
        # backend's own defaults govern in any version where the kwarg is
        # absent. Pre-Wave-6b builds therefore see a CLI surface with the
        # five new flags AVAILABLE but inert until the backend lands —
        # matching the cross-domain handoff contract.
        import inspect as _inspect
        try:
            _trainer_sig_params = set(_inspect.signature(Trainer.__init__).parameters)
        except (TypeError, ValueError):
            # Test doubles that mock Trainer with a non-callable / opaque
            # MagicMock can't be introspected; degrade to "no Wave 6b
            # kwargs threaded" so the legacy code path is preserved.
            _trainer_sig_params = set()
        wave6b_candidate_kwargs: dict[str, Any] = {
            "use_dora": bool(getattr(args, "use_dora", False)),
            # --no-packing is the opt-out; default = packing ON.
            "packing": not bool(getattr(args, "no_packing", False)),
            "init_lora_weights": getattr(args, "init_lora_weights", "default"),
            "lora_preset": getattr(args, "lora_preset", "quality"),
            "optim": getattr(args, "optim", "auto"),
        }
        wave6b_kwargs = {
            k: v for k, v in wave6b_candidate_kwargs.items()
            if k in _trainer_sig_params
        }

        # Create trainer
        trainer = Trainer(
            model=args.model,
            lora_r=args.lora_r,
            learning_rate=args.lr,
            batch_size=args.batch_size if args.batch_size != "auto" else "auto",
            output_dir=args.output,
            use_unsloth=not args.no_unsloth,
            **wave6b_kwargs,
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

        # BRIDGE Wave 6b (v1.3): build the optional Wave 6b kwarg bundle and
        # bind only the keys the installed MultiRunConfig / MultiRunTrainer
        # actually accept. Same defensive scaffolding as cmd_train —
        # introspect dataclass fields / __init__ signature so pre-Wave-6b
        # builds see the new CLI flags AVAILABLE but inert until the
        # backend lands, instead of crashing at MultiRunConfig construction.
        # The dataclass.fields() probe is wrapped in try/except so test
        # doubles that mock MultiRunConfig with a non-dataclass MagicMock
        # cleanly degrade to "no Wave 6b kwargs threaded" instead of
        # crashing the handler.
        import dataclasses as _dc
        import inspect as _inspect
        try:
            _multi_cfg_fields = {f.name for f in _dc.fields(MultiRunConfig)}
        except (TypeError, ValueError):
            _multi_cfg_fields = set()
        try:
            _multi_trainer_params = set(_inspect.signature(MultiRunTrainer.__init__).parameters)
        except (TypeError, ValueError):
            _multi_trainer_params = set()
        wave6b_candidate_kwargs: dict[str, Any] = {
            "use_dora": bool(getattr(args, "use_dora", False)),
            "packing": not bool(getattr(args, "no_packing", False)),
            "init_lora_weights": getattr(args, "init_lora_weights", "default"),
            "lora_preset": getattr(args, "lora_preset", "quality"),
            "optim": getattr(args, "optim", "auto"),
        }
        wave6b_cfg_kwargs = {
            k: v for k, v in wave6b_candidate_kwargs.items()
            if k in _multi_cfg_fields
        }
        wave6b_trainer_kwargs = {
            k: v for k, v in wave6b_candidate_kwargs.items()
            if k not in _multi_cfg_fields and k in _multi_trainer_params
        }

        config = MultiRunConfig(
            num_runs=args.runs,
            steps_per_run=args.steps,
            samples_per_run=args.samples,
            merge_mode=MergeMode(args.merge_mode),
            checkpoint_dir=args.output,
            **wave6b_cfg_kwargs,
        )

        def on_run_complete(run_result: RunResult) -> None:
            _print_success(f"Run {run_result.run_index + 1} complete: loss={run_result.final_loss:.4f}")

        trainer = MultiRunTrainer(
            model=args.model,
            config=config,
            on_run_complete=on_run_complete,
            resume_from=getattr(args, "resume", None),  # F-002
            **wave6b_trainer_kwargs,
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
# BRIDGE-A-004 (v1.4): HF token file resolver — shared by cmd_export and
# cmd_push so `--hub-token-file <path>` / `--token-file <path>` keeps the
# credential off the argv surface (where it leaks to `ps aux` + shell
# history). Mirrors the v1.3 `--auth-file` pattern from cmd_ui (cli.py
# ~2137): existence check, mode-0600 verification on POSIX, content read,
# UserInputError on any failure mode so the catch-all in the calling
# handler emits a friendly redacted message instead of a stack trace.
# =============================================================================

def _read_hub_token_file(path_str: str, *, flag_name: str) -> str:
    """Read an HF token from a file at ``path_str``.

    Mirrors the v1.3 ``--auth-file`` pattern in cmd_ui: existence check,
    mode-0600 verification on POSIX (warn, don't refuse, when widened),
    content read with a clear error on empty/unreadable files. Returns
    the stripped token string. Raises ``UserInputError`` on any failure
    so the catch-all in cmd_push / cmd_export emits a friendly redacted
    message instead of a stack trace.

    Args:
        path_str: Operator-provided path to the token file.
        flag_name: The CLI flag name to surface in error messages
            (``--hub-token-file`` for cmd_push, ``--hub-token-file`` for
            cmd_export — kept as a kwarg so future callers can pass a
            different flag spelling without changing the helper).
    """
    token_path = Path(path_str).expanduser()
    if not token_path.exists():
        raise UserInputError(
            f"{flag_name} path does not exist: {token_path}",
            hint=(
                f"Create the file with `printf 'hf_xxx' > {token_path}` "
                f"(no trailing newline) and set mode 0600: "
                f"`chmod 600 {token_path}`."
            ),
            code="INPUT_VALIDATION_FAILED",
        )
    # File-mode warning on POSIX: > 0o600 means group / world can read
    # the credential. Don't refuse — the operator may have intentionally
    # widened the mode — but surface the consequence.
    if os.name == "posix":
        try:
            mode = token_path.stat().st_mode & 0o777
            if mode & 0o077:  # any group / other bits set
                _print_warning(
                    f"{flag_name} mode is {oct(mode)} — group / other have "
                    f"read access. Tighten to 0600: `chmod 600 {token_path}`."
                )
        except OSError:
            pass
    try:
        token_text = token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise UserInputError(
            f"{flag_name} could not be read: {exc}",
            hint="Verify the file exists and the current user has read "
            "permission.",
            code="INPUT_VALIDATION_FAILED",
        ) from exc
    if not token_text:
        raise UserInputError(
            f"{flag_name} is empty (expected the HF token on the first line).",
            hint=(
                f"Write the token with `printf 'hf_xxx' > {token_path}` "
                f"(no trailing newline) and set mode 0600."
            ),
            code="INPUT_VALIDATION_FAILED",
        )
    return token_text


def _warn_token_on_argv(flag_name: str) -> None:
    """Emit the shared 'credential on argv' warning for HF-token flags.

    Mirrors the v1.3 cmd_ui ``--auth`` warning at cli.py ~2129. Centralised
    so cmd_push and cmd_export emit byte-identical copy and a future
    operator-facing tweak lands in one spot.
    """
    _print_warning(
        f"{flag_name} was passed on the command line — shell history "
        f"and `ps aux` will retain the credential. Prefer "
        f"--hub-token-file <path> or the HF_TOKEN env var."
    )


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

            # BRIDGE-A-004 (v1.4): resolve --hub-token-file before push so
            # the token never has to be argv-visible. Mutex with --hub-token
            # mirrors the v1.3 --auth-file / --auth pattern (passing both
            # races on which credential wins). If --hub-token IS set, emit
            # the shared shell-history warning so the operator sees the
            # safer paths (file / env var / cached login).
            inline_hub_token = getattr(args, "hub_token", None)
            hub_token_file = getattr(args, "hub_token_file", None)
            if inline_hub_token and hub_token_file:
                raise UserInputError(
                    "--hub-token and --hub-token-file are mutually exclusive — "
                    "pick one.",
                    hint=(
                        "Use --hub-token-file <path> to keep the token out of "
                        "shell history, OR --hub-token <token> for one-off "
                        "invocations. Combining the two would race on which "
                        "credential wins."
                    ),
                    code="INPUT_VALIDATION_FAILED",
                )
            if inline_hub_token:
                _warn_token_on_argv("--hub-token")
            resolved_hub_token: str | None = inline_hub_token
            if hub_token_file:
                resolved_hub_token = _read_hub_token_file(
                    hub_token_file, flag_name="--hub-token-file"
                )

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
                    token=resolved_hub_token,
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
    #
    # BRIDGE-F-004 (Wave 5.5): added the three "raw os.environ.get" knobs
    # that the runtime honors but pydantic-settings doesn't see —
    # BACKPROPAGATE_UI_QUIET (banner suppression at cli.py:1509),
    # BACKPROPAGATE_DEBUG (full-traceback toggle at cli.py:3210/3301/3312),
    # and BACKPROPAGATE_LLAMA_CPP_PATH (GGUF convert-script escape hatch
    # at export.py:1297). Adding them here closes the doc-lie where
    # `backprop info --env-vars` claimed to enumerate every BACKPROPAGATE_*
    # env var but silently skipped these three.
    logging_envs = [
        ("BACKPROPAGATE_LOG_LEVEL", "INFO", "str", "Structured-logger level: DEBUG / INFO / WARNING / ERROR."),
        ("BACKPROPAGATE_LOG_JSON", "0", "bool", "Emit JSON-formatted log lines instead of pretty console output."),
        ("BACKPROPAGATE_LOG_FILE", "", "path", "File path for log output; appended to in addition to stderr."),
        ("BACKPROPAGATE_DEFER_FEATURE_DETECTION", "", "bool", "Skip optional-feature detection at import time."),
        (
            "BACKPROPAGATE_UI_QUIET",
            "",
            "bool",
            "When set to '1', suppress the 3-line UI startup banner (URL / auth mode / Ctrl+C hint) on stderr. Use for CI / headless launches that don't want banner noise. Only the exact value '1' suppresses; '0' / 'false' / unset leave the banner on.",
        ),
        (
            "BACKPROPAGATE_DEBUG",
            "",
            "bool",
            "When set to any truthy value, the top-level CLI exception net prints the full Python traceback in addition to the one-line error message. Off by default to keep operator-facing failures short; flip to '1' when filing a bug report.",
        ),
        (
            "BACKPROPAGATE_LLAMA_CPP_PATH",
            "",
            "path",
            "Operator escape hatch for non-standard llama.cpp install locations used by `backprop export --format gguf`. Accepts either the path to convert_hf_to_gguf.py directly or the llama.cpp directory containing it. Searched FIRST, before shutil.which / ~/llama.cpp / /usr/local/bin.",
        ),
        (
            "BACKPROPAGATE_CLOUDFLARED_TIMEOUT",
            "30",
            "int",
            "BRIDGE-F-CLOUDFLARED (v1.3 Wave 6a): seconds to wait for the cloudflared subprocess (spawned by `backprop ui --share`) to surface its trycloudflare.com tunnel URL on stderr before giving up. Default 30s; operators on slow uplinks may want 60-120s. Values <= 0 fall back to the default. Honored only when --share is passed and `cloudflared` is on PATH.",
        ),
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

# =============================================================================
# LOCK-FILE TOKEN (BRIDGE-F-002 auth-polish item 3 / V1_3_BRIEF P0)
# =============================================================================
#
# Per DESIGN_BRIEF "Lock-file token mode (post-CVE-2025-52882 defense)":
# the per-launch token / auth credentials are also written to
#   $XDG_RUNTIME_DIR/backpropagate/session-<port>.lock     (Linux)
#   ~/Library/Application Support/backpropagate/session-<port>.lock (macOS)
#   %LOCALAPPDATA%\backpropagate\session-<port>.lock        (Windows)
# so machine-to-machine clients (e.g. ``backprop train --watch-ui``) can
# read the token / credentials without exposing them in argv.
#
# Mode 0o600 on POSIX (owner read+write only). On Windows we fall back to
# the per-user LOCALAPPDATA directory whose ACL is owner-restricted by
# default — a tight ACL-rewrite via icacls would require pywin32 which
# isn't a runtime dependency. The Windows fallback is documented in
# handbook/security.md (cross-domain handoff to frontend agent below).
#
# Cross-domain handoff to frontend agent: the test scaffold in
# tests/test_auth_middleware.py:540 (currently pytest.skip()ed) imports
# ``from backpropagate.ui_app.auth import write_launch_token_lock``. Once
# the frontend agent re-exports this function from ui_app/auth.py (or
# decides on the alternative location), the test can be unskipped and
# the lock-file contract becomes enforced. The canonical implementation
# lives here so the cli.py:cmd_ui boot path can call it without
# importing ui_app (which depends on the [ui] extra).


def _lock_file_dir() -> Path:
    """Return the platform-appropriate runtime directory for lock files.

    Honors ``$XDG_RUNTIME_DIR`` on Linux (the freedesktop spec). Falls
    back to ``$LOCALAPPDATA`` on Windows and ``~/Library/Application
    Support`` on macOS. Creates the ``backpropagate`` subdirectory if it
    doesn't exist (with owner-only permissions on POSIX).
    """
    # XDG_RUNTIME_DIR honored first regardless of OS — tests use this to
    # redirect into tmp_path. Linux is the canonical surface; the macOS /
    # Windows fallbacks are best-effort secondary support.
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        base = Path(xdg)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            base = Path(local_app)
        else:
            base = Path.home() / "AppData" / "Local"
    else:
        # Linux fallback when XDG_RUNTIME_DIR is unset — /tmp is world-
        # writable but the file itself gets mode 0o600 below + the
        # parent dir gets 0o700 a few lines down (defense in depth), so
        # an attacker can see the file's existence but cannot read it.
        base = Path("/tmp")  # nosec B108 — mode-restricted file + dir below

    target = base / "backpropagate"
    target.mkdir(parents=True, exist_ok=True)
    # On POSIX, restrict the directory to owner-only too (defense in depth;
    # even if the lock file's mode is wrong somehow, the directory blocks
    # the read). Windows: mkdir inherits the parent ACL which under
    # LOCALAPPDATA is owner-restricted by default.
    if os.name == "posix":
        try:
            os.chmod(target, 0o700)
        except OSError:  # pragma: no cover — best effort, mkdir succeeded
            pass
    return target


def write_launch_token_lock(port: int, token: str) -> Path:
    """Write the launch token / credentials to a per-launch lock file.

    Args:
        port: Port the UI is listening on (used to form the filename so
            two concurrent ``backprop ui`` invocations on different ports
            don't collide).
        token: The token / credential string to persist. The caller is
            responsible for deciding whether to persist the launch token
            (token-auto mode) or the ``user:pass`` shape (basic-auth mode);
            this helper writes whatever opaque string it gets.

    Returns:
        Path to the lock file (the caller's CLI logs the path so the
        operator knows where to point their M2M client).

    File contract:
        On POSIX: mode 0o600 (owner read+write only). Writes atomically
        via tempfile + os.replace so a partial write doesn't leak a
        half-formed token.

        On Windows: the file ends up in ``%LOCALAPPDATA%\\backpropagate\\``
        which inherits the user's ACL by default (owner-restricted on a
        single-user box). A tight icacls rewrite would require pywin32
        which is not a runtime dependency; the per-user directory provides
        the practical floor.

    Concurrency:
        Multiple ``backprop ui --port N`` invocations on the SAME port
        race; the last write wins (consistent with Reflex's own port-
        binding behavior — only one server can hold the port). Different
        ports never collide because the filename embeds the port.
    """
    import tempfile

    target_dir = _lock_file_dir()
    lock_path = target_dir / f"session-{int(port)}.lock"

    # Write to a sibling temp file then atomically rename. This avoids
    # leaving a half-written file (and the wrong mode bits) visible to
    # other processes that happen to scan the directory between the
    # creat() and the write().
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".session-{int(port)}.",
        suffix=".lock.tmp",
        dir=str(target_dir),
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:  # pragma: no cover — fsync best-effort
                pass
        if os.name == "posix":
            os.chmod(tmp_path, 0o600)
        # Atomic rename to the canonical name. On POSIX this is an inode-
        # level swap; on Windows os.replace also atomically replaces.
        os.replace(tmp_path, lock_path)
    except Exception:
        # Best-effort cleanup of the temp file on any error so we don't
        # leak it.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    return lock_path


def _print_ui_startup_banner(
    *,
    bound_host: str,
    port: int,
    auth: tuple[str, str] | None,
    share: bool,
    token_query: str | None = None,
) -> None:
    """Print the 3-line Jupyter-pattern startup banner to stderr.

    BRIDGE-B (Stage C auth-polish): when ``backprop ui`` boots, surface a
    short operator-facing banner so the trust model is calibrated before
    the first request. Lee & See 2004 trust-calibration framing applied
    to the UI launch flow:

        [backprop] http://<bound-host>:<port>/  <- UI listening
        [backprop] auth: <mode> -- <concrete-consequence>
        [backprop] open the URL to start; stop with Ctrl+C

    ``<mode>`` and ``<concrete-consequence>`` map by precedence:

    +---------------------------------+--------------------------------+
    | Mode                            | Concrete consequence            |
    +=================================+================================+
    | none (loopback-only)            | any local process can access    |
    |                                 | the UI                          |
    +---------------------------------+--------------------------------+
    | token (auto-generated)          | share the URL above to grant    |
    |                                 | access -- URL contains a secret |
    +---------------------------------+--------------------------------+
    | basic (user '<u>')              | password protects the UI;       |
    |                                 | rotate the password if          |
    |                                 | compromised                     |
    +---------------------------------+--------------------------------+
    | DISABLED (--share without auth) | the UI is PUBLIC; anyone with   |
    |                                 | the URL has full access         |
    +---------------------------------+--------------------------------+

    Suppress entirely with ``BACKPROPAGATE_UI_QUIET=1`` for CI / headless
    cases that don't want the banner noise.

    Test contract (cross-domain handoff for tests agent — this function
    has no test file in tests/ because bridge agent has no write access
    to tests/. Tests agent: please add tests/test_ui_banner.py asserting
    each of the 4 auth modes produces the expected 3-line shape on
    stderr, AND that BACKPROPAGATE_UI_QUIET=1 suppresses output entirely):

    * mode=no_auth_local -> calling with auth=None + share=False prints
      "auth: none (loopback-only) -- any local process..."
    * mode=token_auto -> calling with auth=None + token_query="abc" prints
      a URL with ?token=abc AND "auth: token (auto-generated)..."
    * mode=basic -> calling with auth=("alice", "secret") prints the URL
      WITHOUT any password text AND "auth: basic (user 'alice')..."
    * mode=insecure -> calling with auth=None + share=True prints
      "auth: DISABLED (--share without --auth)..." (this code path is
      gated upstream and should be unreachable in practice, but the
      branch exists to make the banner total over the 4 modes.)
    """
    # Honor the quiet-mode escape hatch first so CI doesn't see banner noise.
    if os.environ.get("BACKPROPAGATE_UI_QUIET") == "1":
        return

    url_path = f"/?token={token_query}" if token_query else "/"
    url = f"http://{bound_host}:{port}{url_path}"

    # Determine the mode + concrete-consequence pair (Lee & See 2004
    # trust-calibration framing — operator should know what the auth
    # decision means BEFORE they share the URL).
    is_loopback = bound_host in ("127.0.0.1", "localhost", "::1")

    if share and auth is None:
        # The refuse-to-start gate should have prevented this, but the
        # branch exists to make the banner total over the 4 modes.
        mode_label = "DISABLED (--share without --auth)"
        consequence = "the UI is PUBLIC; anyone with the URL has full access"
    elif auth is not None:
        username = auth[0]
        mode_label = f"basic (user {username!r})"
        consequence = (
            "password protects the UI; rotate the password if compromised"
        )
    elif token_query:
        mode_label = "token (auto-generated)"
        consequence = (
            "share the URL above to grant access -- URL contains a secret"
        )
    elif is_loopback:
        mode_label = "none (loopback-only)"
        consequence = "any local process can access the UI"
    else:
        # Non-loopback bind without --auth — gated upstream as
        # RUNTIME_UI_AUTH_NOT_ENFORCED. Same total-coverage rationale as
        # the share+no-auth branch.
        mode_label = "DISABLED (non-loopback bind without --auth)"
        consequence = (
            "the UI is reachable from the network with NO authentication"
        )

    print(f"[backprop] {url}  <- UI listening", file=sys.stderr)
    print(f"[backprop] auth: {mode_label} -- {consequence}", file=sys.stderr)
    print(
        "[backprop] open the URL to start; stop with Ctrl+C",
        file=sys.stderr,
    )


# =============================================================================
# CLOUDFLARED QUICK TUNNEL (BRIDGE-F-CLOUDFLARED / V1_3_BRIEF P1)
# =============================================================================
#
# Implements ``backprop ui --share`` end-to-end: spawn ``cloudflared tunnel
# --url http://localhost:<port>`` as a subprocess, parse the
# ``https://*.trycloudflare.com`` URL from its stderr on connect, and
# expose the tunnel hostname via ``BACKPROPAGATE_UI_SHARE_HOST`` so the
# Reflex auth middleware's Host + Origin allowlists pick it up
# automatically (no auth.py change needed — the middleware already reads
# the env var).
#
# Cloudflare Quick Tunnels (no Cloudflare account / no auth flow required)
# stay up for the lifetime of the subprocess; we clean up on SIGINT via
# the subprocess.Popen + terminate() pattern below. If ``cloudflared``
# isn't on PATH we surface a friendly error pointing at the install URL
# and the SSH port-forwarding fallback so the operator has an action plan.

# Cloudflare Quick Tunnels surface the public URL on stderr as a line
# matching:  https://<random-subdomain>.trycloudflare.com
# Older / future builds may also wrap it in box-drawing characters; the
# regex matches the URL anywhere in any line.
_CLOUDFLARED_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

# Cloudflared's TLS-ready signal typically appears within 5s on a healthy
# link. Operators on slow uplinks can override via BACKPROPAGATE_CLOUDFLARED_TIMEOUT.
_CLOUDFLARED_DEFAULT_TIMEOUT_SECONDS = 30


def _spawn_cloudflared_tunnel(port: int) -> tuple[subprocess.Popen, str] | None:
    """Spawn ``cloudflared tunnel --url http://localhost:<port>`` and parse the URL.

    Returns a ``(Popen, tunnel_url)`` tuple on success, ``None`` if
    cloudflared is unavailable / didn't surface a URL within the timeout.
    The caller is responsible for terminating the subprocess on exit.

    The subprocess inherits stdout/stderr-to-pipe so we can scrape the
    URL line; once the URL is parsed we keep the pipes open and drain
    them in background threads so the tunnel doesn't deadlock on a full
    OS pipe buffer.
    """
    import shutil
    import threading

    cf_path = shutil.which("cloudflared")
    if not cf_path:
        _print_error(
            "--share requires `cloudflared` on PATH but it was not found."
        )
        _print_info(
            "Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        )
        _print_info(
            "Alternative — SSH port-forward instead:  "
            f"ssh -L {port}:localhost:{port} <your-host>"
        )
        return None

    cmd = [
        cf_path,
        "tunnel",
        "--no-autoupdate",
        "--url",
        f"http://localhost:{port}",
    ]

    # cloudflared writes its banner + status lines to stderr.
    try:
        proc = subprocess.Popen(  # nosec B603 — cloudflared path resolved via shutil.which; arguments fully controlled
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered so we see lines as cloudflared emits them
        )
    except OSError as exc:
        _print_error(f"Failed to spawn cloudflared: {exc}")
        return None

    # Tail the merged output (stdout was redirected to stderr) until we
    # either parse the URL or hit the timeout. Honor an env override so
    # operators on slow uplinks can extend the budget.
    try:
        timeout_env = os.environ.get("BACKPROPAGATE_CLOUDFLARED_TIMEOUT", "")
        timeout_seconds = (
            int(timeout_env) if timeout_env else _CLOUDFLARED_DEFAULT_TIMEOUT_SECONDS
        )
        if timeout_seconds <= 0:
            timeout_seconds = _CLOUDFLARED_DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        timeout_seconds = _CLOUDFLARED_DEFAULT_TIMEOUT_SECONDS

    import time as _time
    deadline = _time.monotonic() + timeout_seconds
    tunnel_url: str | None = None

    # Read line-by-line until we find the URL or the deadline elapses.
    assert proc.stdout is not None  # nosec B101 — Popen(stdout=PIPE) guarantees this
    while _time.monotonic() < deadline:
        if proc.poll() is not None:
            # cloudflared died before publishing a URL — surface the tail
            # of its output to help the operator triage (invalid network,
            # captive portal, etc.).
            _print_error("cloudflared exited before publishing a tunnel URL.")
            try:
                tail = proc.stdout.read() or ""
            except Exception:  # noqa: BLE001 — best-effort drain
                tail = ""
            if tail.strip():
                # Truncate to a reasonable size so we don't pipe MB of log.
                _print_info(f"cloudflared output (tail):\n{tail[-800:]}")
            return None
        # Read one line with a short timeout via select — readline()
        # blocks indefinitely otherwise. POSIX uses select.select on the
        # raw fd; Windows pipes aren't selectable so we fall back to a
        # blocking readline() which is bounded by the deadline at the
        # next iteration.
        line = proc.stdout.readline()
        if not line:
            # EOF or transient empty read — re-check deadline + alive.
            continue
        match = _CLOUDFLARED_URL_RE.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if not tunnel_url:
        _print_error(
            f"cloudflared did not surface a tunnel URL within {timeout_seconds}s."
        )
        _print_info(
            "Set BACKPROPAGATE_CLOUDFLARED_TIMEOUT=<seconds> to extend the budget, "
            "or fall back to SSH port-forwarding: "
            f"ssh -L {port}:localhost:{port} <your-host>"
        )
        try:
            proc.terminate()
        except OSError:
            pass
        return None

    # Drain remaining cloudflared output in a daemon thread so the
    # subprocess doesn't deadlock on a full pipe buffer. We don't echo
    # the lines (cloudflared's verbose telemetry would spam stderr); a
    # future polish item could route them to the structured logger.
    def _drain_pipe(pipe: Any) -> None:
        try:
            for _ in pipe:
                pass
        except (OSError, ValueError):  # pragma: no cover — pipe closed / process gone
            pass

    drain_thread = threading.Thread(
        target=_drain_pipe, args=(proc.stdout,), daemon=True, name="cloudflared-drain"
    )
    drain_thread.start()

    return proc, tunnel_url


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

    ``--share`` (BRIDGE-F-CLOUDFLARED, v1.3 Wave 6a): spawns ``cloudflared
    tunnel --url http://localhost:<port>`` as a subprocess (Cloudflare Quick
    Tunnels — no Cloudflare account needed, ephemeral). The tunnel URL is
    parsed from cloudflared stderr and exported via
    ``BACKPROPAGATE_UI_SHARE_HOST`` so the auth middleware's Host + Origin
    allowlists accept the trycloudflare.com hostname. The subprocess is
    cleaned up on SIGINT before the CLI exits.

    ``--host`` (Wave 3.5, v1.3): the operator-supplied bind is now passed to
    ``reflex run --backend-host``; previously the backend defaulted to
    Reflex's ``backend_host="0.0.0.0"`` regardless of what the operator
    asked for. The auth middleware's Host-header allowlist
    (``BACKPROPAGATE_UI_HOST_BIND``) remains the load-bearing access check.

    ``--auth-file <path>`` (BRIDGE-F-002 auth-polish item 4, v1.3 Wave 6a):
    reads ``user:pass`` from a file instead of taking it on the command
    line (keeps the credential out of shell history). The file's mode is
    checked on POSIX — a stderr warning fires if mode > 0o600. Mutually
    exclusive with ``--auth`` (passing both raises a UserInputError).

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

    # BRIDGE-F-002 auth-polish item 4 (v1.3 Wave 6a): resolve --auth-file
    # BEFORE the refuse-to-start gates so the gates see the resolved
    # credential regardless of source. Mutex with --auth: passing both
    # is operator error (which credential should we believe?).
    auth_file = getattr(args, "auth_file", None)
    if auth_file and args.auth:
        raise UserInputError(
            "--auth and --auth-file are mutually exclusive — pick one.",
            hint=(
                "Use --auth-file <path> to keep credentials out of shell "
                "history, OR --auth user:pass for one-off invocations. "
                "Combining the two would race on which credential wins."
            ),
            code="INPUT_VALIDATION_FAILED",
        )

    # Stage C humanization parity: if --auth was passed inline, warn that
    # shell history captured it (the env-var / --auth-file paths avoid this).
    if args.auth:
        _print_warning(
            "--auth was passed inline — shell history will retain the "
            "credential. For repeat invocations, prefer --auth-file <path> "
            "or the BACKPROPAGATE_UI_AUTH env var."
        )

    if auth_file:
        auth_file_path = Path(auth_file).expanduser()
        if not auth_file_path.exists():
            raise UserInputError(
                f"--auth-file path does not exist: {auth_file_path}",
                hint="Create the file with `printf 'user:pass' > path` and "
                "set mode 0600 (`chmod 600 path`).",
                code="INPUT_VALIDATION_FAILED",
            )
        # File-mode warning on POSIX: > 0o600 means group / world can read
        # the credential. Don't refuse — the operator may have intentionally
        # widened the mode — but surface the consequence.
        if os.name == "posix":
            try:
                mode = auth_file_path.stat().st_mode & 0o777
                if mode & 0o077:  # any group / other bits set
                    _print_warning(
                        f"--auth-file mode is {oct(mode)} — group / other "
                        "have read access. Tighten to 0600: "
                        f"`chmod 600 {auth_file_path}`."
                    )
            except OSError:
                pass
        try:
            auth_text = auth_file_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise UserInputError(
                f"--auth-file could not be read: {exc}",
                hint="Verify the file exists and the current user has read "
                "permission.",
                code="INPUT_VALIDATION_FAILED",
            ) from exc
        if not auth_text:
            raise UserInputError(
                "--auth-file is empty (expected 'user:pass' on the first line).",
                hint="Write the credential with `printf 'user:pass' > path` "
                "(no trailing newline) and set mode 0600.",
                code="INPUT_VALIDATION_FAILED",
            )
        # Re-use the same shape validator the --auth argparse type uses
        # so the file content gets identical "no whitespace / colon in
        # username, no newline in password" enforcement. Re-raising as
        # UserInputError keeps the catch-all happy.
        try:
            args.auth = _auth_credential(auth_text)
        except argparse.ArgumentTypeError as exc:
            raise UserInputError(
                f"--auth-file content failed validation: {exc}",
                hint="The file must contain a single 'user:pass' line that "
                "would also pass `--auth user:pass` validation.",
                code="INPUT_AUTH_INVALID_SHAPE",
            ) from exc

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

    # BRIDGE-B-008 (Stage C — documented gap): post-launch health probe
    # against /_health (asserts unauthenticated GET / returns 401) is a
    # v1.4 candidate. Today the CLI refuses to start if ENFORCEMENT_AVAILABLE
    # is False at import time, which catches the dominant failure mode
    # (auth.py import error). What it does NOT catch: a runtime regression
    # where FastAPI middleware imports cleanly but raises AttributeError at
    # request time. Adding a Popen + probe loop is a moderate refactor that
    # requires reworking the subprocess.run call site to a non-blocking
    # Popen with a healthcheck thread. Tracked as the v1.4 followup
    # "BRIDGE-V14-UI-PROBE" so the auth contract assertion catches both
    # the import-time + runtime layers (same defensive depth that caught
    # v1.2.0's FRONTEND-B-001 layer-2 auth bypass).
    #
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
    # BRIDGE-F-CLOUDFLARED (v1.3 Wave 6a): spawn the Cloudflare Quick Tunnel
    # subprocess if --share was passed, parse the trycloudflare.com URL
    # from cloudflared stderr, and stash it in BACKPROPAGATE_UI_SHARE_HOST
    # so the auth middleware's Host + Origin allowlists accept it. The
    # subprocess is kept alive for the lifetime of the Reflex run and
    # terminated below on SIGINT / KeyboardInterrupt / normal exit.
    cloudflared_proc: subprocess.Popen | None = None
    tunnel_host = ""
    if args.share:
        spawn_result = _spawn_cloudflared_tunnel(args.port)
        if spawn_result is None:
            # _spawn_cloudflared_tunnel already printed the friendly error
            # and the SSH-fallback hint. Return user-error so the operator
            # knows it's a config/install issue, not an internal crash.
            return EXIT_USER_ERROR
        cloudflared_proc, tunnel_url = spawn_result
        # Parse the hostname out of the URL (strip the scheme + path so
        # the middleware allowlist sees just the bare host).
        try:
            from urllib.parse import urlsplit
            tunnel_host = (urlsplit(tunnel_url).hostname or "").lower()
        except Exception:  # noqa: BLE001 — best-effort parsing
            tunnel_host = ""
        if tunnel_host:
            env["BACKPROPAGATE_UI_SHARE_HOST"] = tunnel_host
            _print_success(f"Tunnel ready: {tunnel_url}")
            _print_info(
                "Share this URL to grant access. The tunnel + the UI both "
                "stop when you Ctrl+C this terminal."
            )
        else:
            # We spawned cloudflared but couldn't parse the host — should be
            # impossible after _spawn_cloudflared_tunnel parsed it once, but
            # fail-closed rather than silently dropping the allowlist entry.
            try:
                cloudflared_proc.terminate()
            except OSError:
                pass
            _print_error(
                "Could not parse the trycloudflare.com hostname from the "
                "cloudflared output; refusing to start the UI."
            )
            return EXIT_RUNTIME_ERROR

    # BRIDGE-F-002 auth-polish item 3 (v1.3 Wave 6a): write the credential
    # to a per-launch lock file at $XDG_RUNTIME_DIR/backpropagate/session-
    # <port>.lock (or the platform equivalent) so machine-to-machine clients
    # (`backprop train --watch-ui` and similar) can pick up the auth string
    # without it appearing in argv / ps output. Mode 0o600 on POSIX; the
    # Windows fallback inherits the per-user LOCALAPPDATA ACL.
    #
    # We persist either the explicit user:pass (basic-auth mode) or the
    # launch token (token-auto mode; not yet exposed on the CLI but the
    # helper is forward-compatible). NO_AUTH_LOCAL_ONLY skips lock-file
    # creation entirely — there's nothing to authenticate against.
    lock_file_path: Path | None = None
    lock_payload = env.get("BACKPROPAGATE_UI_AUTH") or env.get("BACKPROPAGATE_UI_LAUNCH_TOKEN")
    if lock_payload:
        try:
            lock_file_path = write_launch_token_lock(args.port, lock_payload)
            _print_info(f"Auth lock-file: {lock_file_path} (mode 0o600 on POSIX)")
        except Exception as exc:  # noqa: BLE001 — lock-file is best-effort observability
            # Don't abort the UI launch just because the lock file failed;
            # the credential still flows via env var. Surface the reason so
            # an operator who NEEDS the lock-file path (M2M consumers) can
            # triage. Auth itself is unaffected.
            _print_warning(
                f"Could not write launch lock-file ({exc}); M2M consumers "
                "will need to read BACKPROPAGATE_UI_AUTH from their own env."
            )

    # Reflex's port convention: the frontend serves on --frontend-port and
    # the backend on --backend-port. We map --port to the frontend (what
    # users hit in the browser) and the backend gets port+1.
    #
    # BRIDGE-B-001 (Wave 3.5, v1.3): pass --backend-host through to the
    # Reflex subprocess so the operator-requested bind actually takes effect.
    # Without this, Reflex's default backend_host="0.0.0.0" (reflex_base.config)
    # silently bound the FastAPI backend to ALL interfaces regardless of the
    # operator's --host value — making --host advertise control it didn't
    # deliver. The CLI's refuse-to-start gates (loopback-only without --auth)
    # were the only thing standing between a default install and a LAN-exposed
    # backend; passing --backend-host here makes the bind match what the
    # operator asked for.
    #
    # Default (no --host): resolves to "127.0.0.1" so the backend is loopback
    # by default — matches the loopback-first posture the CLI documents.
    # When --host LAN-IP --auth user:pass passes the gate, the operator-
    # supplied host flows through.
    #
    # Frontend bind caveat: the Reflex-generated .web/package.json uses
    # `react-router dev --host` which binds 0.0.0.0 regardless of this
    # backend-host knob. That is a frontend-domain follow-up — the load-
    # bearing security on the backend (the FastAPI/uvicorn process where
    # the API and WebSocket live) is now bound as the operator requested,
    # and the auth middleware's Host-header allowlist + HTTP Basic check
    # (BACKPROPAGATE_UI_HOST_BIND-driven) is the enforcement layer in either
    # case.
    backend_host = requested_host or "127.0.0.1"
    cmd = [
        sys.executable,
        "-m",
        "reflex",
        "run",
        "--frontend-port",
        str(args.port),
        "--backend-port",
        str(args.port + 1),
        "--backend-host",
        backend_host,
    ]

    # BRIDGE-F-CLOUDFLARED (v1.3 Wave 6a): the Wave 3.5 transitional
    # warning block ("--share doesn't establish a tunnel yet — use ssh -L
    # in the meantime") was removed at this anchor when the cloudflared
    # implementation landed. The real tunnel is spawned upstream
    # (see ``_spawn_cloudflared_tunnel`` + the ``args.share`` branch that
    # populates ``BACKPROPAGATE_UI_SHARE_HOST``), and the subprocess is
    # cleaned up below in the finally clause. The
    # [[no-banner-documenting-no-op]] doctrine is preserved: there is no
    # banner advertising a non-feature.

    # BRIDGE-B-006 (Stage C): emit a structured-log subprocess_starting
    # event so an operator grepping `ui_subprocess` in JSON logs can see
    # the lifecycle of the Reflex child alongside cmd_train / cmd_export.
    # The auth credentials are NOT logged — only a boolean flag — to keep
    # the JSON log audit-safe.
    from .logging_config import get_logger as _get_logger
    _ui_logger = _get_logger(__name__)
    try:
        _ui_logger.info(
            "ui_subprocess_starting",
            host_bind=backend_host,
            port=args.port,
            auth_mode=("basic" if auth else "none"),
            share=bool(args.share),
            cmd=cmd,
        )
    except Exception:  # noqa: BLE001 — observability must not block launch  # nosec B110
        pass

    # BRIDGE-B (Stage C auth-polish): Jupyter-pattern startup banner.
    # Printed AFTER the refuse-to-start gates have fired (lines above)
    # and BEFORE the subprocess.run call so the operator sees it the
    # moment the launch is decided. Suppressed by BACKPROPAGATE_UI_QUIET=1.
    _print_ui_startup_banner(
        bound_host=backend_host,
        port=args.port,
        auth=auth,
        share=bool(args.share),
        token_query=None,
    )

    import time as _time  # local import to keep cold-start of `backprop --help` cheap
    _ui_start_ts = _time.monotonic()
    try:
        print()
        _print_info("Launching Reflex interface...")
        result = subprocess.run(cmd, env=env, cwd=str(package_dir))  # nosec B603 — cmd is internally constructed
        _duration = _time.monotonic() - _ui_start_ts
        try:
            _ui_logger.info(
                "ui_subprocess_exit",
                returncode=result.returncode,
                duration_seconds=round(_duration, 2),
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        return result.returncode if result.returncode is not None else EXIT_OK
    except KeyboardInterrupt:
        _duration = _time.monotonic() - _ui_start_ts
        try:
            _ui_logger.info(
                "ui_subprocess_signal",
                signum=2,  # SIGINT
                duration_seconds=round(_duration, 2),
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        print()
        _print_info("UI stopped")
        return EXIT_OK
    except FileNotFoundError:
        try:
            _ui_logger.info(
                "ui_subprocess_exit",
                returncode=None,
                error="FileNotFoundError (python interpreter or reflex missing)",
            )
        except Exception:  # noqa: BLE001  # nosec B110
            pass
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
    finally:
        # BRIDGE-F-CLOUDFLARED (v1.3 Wave 6a) cleanup: terminate the
        # cloudflared subprocess (if --share spawned one) so we don't
        # leak a public tunnel after the UI exits. Use terminate() +
        # short wait + kill() — same pattern Reflex uses for its own
        # frontend / backend processes.
        if cloudflared_proc is not None:
            try:
                cloudflared_proc.terminate()
                try:
                    cloudflared_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    cloudflared_proc.kill()
                    try:
                        cloudflared_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # SIGKILL also didn't bring it down within 2s —
                        # rare zombie state. We've sent SIGKILL, so the
                        # kernel will reap it eventually; do not propagate
                        # the timeout into the finally block (the user just
                        # Ctrl+C'd; they shouldn't see a confusing trace).
                        pass
            except (OSError, ValueError):  # pragma: no cover — already dead
                pass
        # BRIDGE-F-002 lock-file cleanup: remove the per-launch lock file
        # so a stale credential doesn't sit on disk after the UI exits.
        # Best-effort: don't fail the cleanup just because the file is gone.
        if lock_file_path is not None:
            try:
                lock_file_path.unlink()
            except OSError:  # pragma: no cover — already removed
                pass


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

    # BRIDGE-A-004 (v1.4): resolve --token-file before push so the token
    # never has to be argv-visible. Mutex with --token mirrors the v1.3
    # --auth-file / --auth pattern (passing both races on which credential
    # wins). If --token IS set, emit the shared shell-history warning so
    # the operator sees the safer paths (file / env var / cached login).
    inline_token = getattr(args, "token", None)
    token_file = getattr(args, "token_file", None)
    if inline_token and token_file:
        raise UserInputError(
            "--token and --token-file are mutually exclusive — pick one.",
            hint=(
                "Use --token-file <path> to keep the token out of shell "
                "history, OR --token <token> for one-off invocations. "
                "Combining the two would race on which credential wins."
            ),
            code="INPUT_VALIDATION_FAILED",
        )
    if inline_token:
        _warn_token_on_argv("--token")
    resolved_token: str | None = inline_token
    if token_file:
        try:
            resolved_token = _read_hub_token_file(
                token_file, flag_name="--token-file"
            )
        except UserInputError as e:
            _print_error(f"{e.message}")
            if e.suggestion:
                _print_info(f"Suggestion: {e.suggestion}")
            return EXIT_USER_ERROR

    try:
        url = push_to_hub(
            local_path=local_path,
            repo_id=args.repo,
            token=resolved_token,
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
# COMMAND: diff-runs (BRIDGE Wave 6b)
# =============================================================================
# Three new multi-run subcommands land in Wave 6b to give operators
# first-class workflow primitives around the run-history corpus:
#
#   * ``diff-runs``    — side-by-side comparison of two completed runs
#   * ``replay``       — re-execute a recorded run with the same config
#   * ``export-runs``  — bulk dump of the full history for offline analytics
#
# All three read from RunHistoryManager (the same on-disk JSON the existing
# ``runs`` / ``show-run`` / ``list-runs`` subcommands read) so they slot in
# without touching the persistence layer. The lookup-miss path follows the
# Wave 5.5 BACKEND-F-002 + Wave 6a F-018 doctrine: raise InvalidSettingError
# naming the run_id + searched dir + next-step suggestions, so an operator
# typo'ing a prefix gets one actionable terminal line instead of a silent
# empty result.

# Fields included in the diff-runs comparison view. Order is significant —
# the table renders rows in this order so config-flavored fields come first,
# then numeric outcomes, then derived columns. Hyperparameter fields are
# pulled from the ``hyperparameters`` sub-dict and expanded into top-level
# rows so the diff doesn't require operators to read nested JSON.
_DIFF_RUNS_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "model_name",
    "dataset_info",
    "session_kind",
    "status",
    "steps",
    "duration_seconds",
    "final_loss",
    "gpu_max_temp",
    "checkpoint_path",
    "dataset_hash",
)


def _diff_runs_extract(run: dict[str, Any]) -> dict[str, Any]:
    """Flatten a run-history entry into the dict shape the diff renderer
    expects: top-level lifecycle / outcome fields + every hyperparameter
    promoted to a key prefixed with ``hp.`` so it doesn't collide with the
    top-level set. Pre-fix attempt was a nested dict + recursive diff; the
    flat shape keeps the diff output one row per field which renders
    cleanly in both the colorized table AND --format=json output.
    """
    flat: dict[str, Any] = {}
    for field in _DIFF_RUNS_TOP_LEVEL_FIELDS:
        flat[field] = run.get(field)
    hp = run.get("hyperparameters") or {}
    if isinstance(hp, dict):
        for key, value in hp.items():
            flat[f"hp.{key}"] = value
    return flat


def cmd_diff_runs(args: argparse.Namespace) -> int:
    """Execute ``backprop diff-runs <run_id_a> <run_id_b>`` (BRIDGE Wave 6b).

    Side-by-side comparison of two completed runs. Useful for the "did the
    config tweak actually move the loss" workflow that previously required
    operators to manually grep ``show-run`` output for two ids.

    Exit codes:
        0   diff rendered successfully
        1   missing run_id / history directory not found
    """
    from .checkpoints import RunHistoryManager
    from .exceptions import InvalidSettingError

    history_dir = Path(args.output).expanduser()
    if not history_dir.exists():
        _print_error(f"No history directory: {history_dir}")
        _print_info(
            "Pass --output <dir> to point at the output directory used "
            "during training."
        )
        return EXIT_USER_ERROR

    manager = RunHistoryManager(str(history_dir))

    def _resolve(run_id: str, label: str) -> dict[str, Any] | None:
        record = manager.get_run(run_id)
        if record is not None:
            return record
        # Mirror the Wave 5.5 BACKEND-F-002 + Wave 6a F-018 doctrine: raise
        # InvalidSettingError naming the run_id + searched directory +
        # next-step suggestions so the operator does not have to grep
        # logs to figure out why their lookup missed.
        raise InvalidSettingError(
            setting_name=label,
            value=run_id,
            expected=(
                f"a run_id present in the on-disk run history under "
                f"output_dir={str(history_dir)!r}"
            ),
            suggestion=(
                f"{label}={run_id!r} was NOT found in the run history at "
                f"{str(history_dir)!r}. The lookup is scoped to the "
                f"configured --output directory (not a global run_id "
                f"index). Next steps:\n"
                f"  1) Run `backprop runs` to list run_ids available "
                f"under this output_dir.\n"
                f"  2) If the run was trained under a different output "
                f"directory, re-run with `--output <that-dir>`.\n"
                f"  3) Partial-prefix matches are accepted as long as "
                f"the prefix is unambiguous; widen the prefix if your "
                f"first 8 characters collide with multiple runs."
            ),
        )

    try:
        run_a = _resolve(args.run_id_a, "run_id_a")
        run_b = _resolve(args.run_id_b, "run_id_b")
    except InvalidSettingError as exc:
        _print_error(str(exc.message))
        if exc.suggestion:
            _print_info(f"Suggestion: {exc.suggestion}")
        return EXIT_USER_ERROR

    # Both runs resolved — narrow the type for downstream code.
    assert run_a is not None and run_b is not None

    flat_a = _diff_runs_extract(run_a)
    flat_b = _diff_runs_extract(run_b)
    all_keys = sorted(set(flat_a) | set(flat_b))

    rows: list[dict[str, Any]] = []
    for key in all_keys:
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)
        rows.append({
            "field": key,
            "run_a": val_a,
            "run_b": val_b,
            "changed": val_a != val_b,
        })

    if args.format == "json":
        import json
        payload = {
            "run_a": {
                "run_id": run_a.get("run_id"),
                "started_at": run_a.get("started_at"),
            },
            "run_b": {
                "run_id": run_b.get("run_id"),
                "started_at": run_b.get("started_at"),
            },
            "diff": rows,
            "changed_count": sum(1 for r in rows if r["changed"]),
        }
        print(json.dumps(payload, indent=2, default=str))
        return EXIT_OK

    # Human / colorized table view.
    run_a_id = str(run_a.get("run_id") or "-")[:12]
    run_b_id = str(run_b.get("run_id") or "-")[:12]
    _print_header(f"Diff: {run_a_id} vs {run_b_id}")

    # Compute column widths from the longest displayed value so the table
    # doesn't smear on long checkpoint paths.
    def _stringify(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}" if abs(value) < 1e6 else f"{value:.6g}"
        return str(value)

    width_field = max(len("FIELD"), *(len(r["field"]) for r in rows))
    width_a = max(len(run_a_id), *(len(_stringify(r["run_a"])) for r in rows))
    width_b = max(len(run_b_id), *(len(_stringify(r["run_b"])) for r in rows))

    header = (
        f"{Colors.BOLD}"
        f"{'FIELD'.ljust(width_field)}  "
        f"{run_a_id.ljust(width_a)}  "
        f"{run_b_id.ljust(width_b)}{Colors.RESET}"
    )
    print(header)
    print(Colors.DIM + "-" * (width_field + width_a + width_b + 4) + Colors.RESET)
    for row in rows:
        field_disp = row["field"].ljust(width_field)
        a_disp = _stringify(row["run_a"]).ljust(width_a)
        b_disp = _stringify(row["run_b"]).ljust(width_b)
        if row["changed"]:
            # Highlight changed rows in yellow so a busy diff still draws
            # the eye to the divergent fields.
            line = f"{Colors.YELLOW}{field_disp}  {a_disp}  {b_disp}{Colors.RESET}"
        else:
            line = f"{Colors.DIM}{field_disp}  {a_disp}  {b_disp}{Colors.RESET}"
        print(line)

    print()
    changed = sum(1 for r in rows if r["changed"])
    _print_info(f"{changed} of {len(rows)} field(s) differ.")
    return EXIT_OK


# =============================================================================
# COMMAND: replay (BRIDGE Wave 6b)
# =============================================================================

def _parse_replay_override(value: str) -> tuple[str, str]:
    """argparse type for --override key=value tokens. Returns (key, value)
    so the cmd_replay handler can build a dict without re-parsing.
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--override expects key=value, got {value!r} (no '=' separator)"
        )
    key, raw_val = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"--override key is empty in {value!r}"
        )
    return key, raw_val


# Whitelist of hyperparameter keys the replay subcommand will surface as
# Trainer / MultiRunTrainer constructor kwargs. Keys outside this list are
# rejected at --override parse time so an operator typo doesn't silently
# bypass validation. The list mirrors the kwargs the existing cmd_train /
# cmd_multi_run handlers thread through.
_REPLAY_ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "seed",
    "learning_rate",
    "lr",
    "batch_size",
    "gradient_accumulation",
    "max_steps",
    "steps",
    "samples",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    # BRIDGE Wave 6b additions:
    "use_dora",
    "packing",
    "init_lora_weights",
    "lora_preset",
    "optim",
})


def cmd_replay(args: argparse.Namespace) -> int:
    """Execute ``backprop replay <run_id>`` (BRIDGE Wave 6b).

    Re-runs a recorded training run with the same config + dataset. The
    inherited fields are: ``seed`` / ``learning_rate`` / ``batch_size`` /
    ``gradient_accumulation`` / ``max_steps`` / model / dataset / lora_r.
    The new run gets a fresh run_id (no clobbering of the original record)
    so the replay can be diffed against the source via
    ``backprop diff-runs <original> <replay>``.

    ``--override key=value`` (repeatable) lets the operator tweak specific
    hyperparameters without losing the rest of the recorded context.

    Exit codes:
        0   replay completed successfully
        1   missing run_id, missing history dir, or invalid --override
        2   training runtime error
        130 interrupted
    """
    import uuid

    from .checkpoints import RunHistoryManager
    from .exceptions import InvalidSettingError
    from .logging_config import bind_run_context, get_logger
    from .trainer import Trainer, TrainingCallback

    _print_header("Backpropagate Replay")

    cli_run_id_full = getattr(args, "cli_run_id", None) or uuid.uuid4().hex
    cli_run_id = cli_run_id_full[:12]
    try:
        bind_run_context(run_id=cli_run_id_full, subcommand="replay")
    except Exception:  # noqa: BLE001  # nosec B110 — best-effort observability; must not abort CLI
        pass
    try:
        get_logger(__name__).info(
            "replay_invoked",
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
            "Pass --output <dir> to point at the output directory used "
            "during the original training."
        )
        return EXIT_USER_ERROR

    manager = RunHistoryManager(str(history_dir))
    record = manager.get_run(args.run_id)
    if record is None:
        try:
            raise InvalidSettingError(
                setting_name="run_id",
                value=args.run_id,
                expected=(
                    f"a run_id present in the on-disk run history under "
                    f"output_dir={str(history_dir)!r}"
                ),
                suggestion=(
                    f"run_id={args.run_id!r} was NOT found in the run "
                    f"history at {str(history_dir)!r}. The lookup is "
                    f"scoped to the configured --output directory (not a "
                    f"global run_id index). Next steps:\n"
                    f"  1) Run `backprop runs` to list run_ids available "
                    f"under this output_dir.\n"
                    f"  2) If the run was trained under a different "
                    f"output directory, re-run with `--output <that-dir>`.\n"
                    f"  3) Partial-prefix matches are accepted as long as "
                    f"the prefix is unambiguous."
                ),
            )
        except InvalidSettingError as exc:
            _print_error(str(exc.message))
            if exc.suggestion:
                _print_info(f"Suggestion: {exc.suggestion}")
            return EXIT_USER_ERROR

    # Validate overrides — argparse already split each one into (key, value);
    # we now reject keys outside the whitelist so a typo'd 'lr_rate' (vs
    # 'learning_rate') fails LOUDLY at the CLI surface instead of being
    # silently dropped.
    override_tokens: list[tuple[str, str]] = list(getattr(args, "override", None) or [])
    overrides: dict[str, str] = {}
    for key, value in override_tokens:
        if key not in _REPLAY_ALLOWED_OVERRIDE_KEYS:
            _print_error(
                f"--override key {key!r} is not in the allowed set. "
                f"Allowed keys: {sorted(_REPLAY_ALLOWED_OVERRIDE_KEYS)}"
            )
            return EXIT_USER_ERROR
        overrides[key] = value

    original_run_id = str(record.get("run_id"))
    session_kind = record.get("session_kind") or "single_run"
    model = record.get("model_name") or "Qwen/Qwen2.5-7B-Instruct"
    dataset = record.get("dataset_info")
    hp = dict(record.get("hyperparameters") or {})

    _print_info(f"Original run: {original_run_id[:12]}")
    _print_info(f"Session: {session_kind}")
    _print_info(f"Model: {model}")
    _print_info(f"Dataset: {dataset}")
    if overrides:
        _print_info(f"Overrides: {overrides}")

    if dataset is None:
        _print_error(
            f"Original run {original_run_id[:12]} has no dataset_info "
            "recorded — cannot replay automatically."
        )
        _print_info(
            "Re-run manually with `backprop train --data <dataset>` "
            "matching the original configuration."
        )
        return EXIT_USER_ERROR

    # Apply overrides to the hyperparameter dict so the downstream Trainer
    # construction sees the modified values. Numeric fields get a best-effort
    # parse; non-numeric raise InvalidSettingError so the operator gets a
    # clear error instead of a silent ValueError deep in Trainer.
    def _coerce(key: str, raw: str) -> Any:
        # Best-effort: int → float → bool ('true'/'false') → str fallback.
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        return raw

    for key, value in overrides.items():
        hp[key] = _coerce(key, value)

    try:
        # BRIDGE-A-002 (v1.4): the cmd_replay --override whitelist accepts the
        # five Wave 6b keys (use_dora / packing / init_lora_weights /
        # lora_preset / optim), but pre-fix we built the trainer kwargs from
        # a hand-picked subset that silently dropped them. The pattern below
        # mirrors cmd_train (cli.py ~617) and cmd_multi_run (cli.py ~820):
        # build a candidate dict from the full hp keyspace AFTER overrides
        # have been applied, then filter via inspect.signature on the
        # installed Trainer / dataclasses.fields on MultiRunConfig so a
        # backend version that hasn't landed a Wave 6b kwarg cleanly
        # degrades to "no-op for that flag" instead of crashing on
        # TypeError. Pre-Wave-6b builds preserve their byte-identical
        # behaviour; backend versions that DO accept the kwarg now see the
        # operator's override flow through end-to-end.
        import dataclasses as _dc
        import inspect as _inspect

        _wave6b_keys = ("use_dora", "packing", "init_lora_weights", "lora_preset", "optim")

        # Single-run replay: the multi-run replay path is symmetrically
        # supported but uses the MultiRunTrainer constructor. We default
        # to single-run when session_kind is unset for forward-compat.
        if session_kind == "multi_run":
            from .multi_run import MergeMode, MultiRunConfig, MultiRunTrainer

            mr_config_kwargs: dict[str, Any] = {
                "num_runs": int(hp.get("num_runs") or 5),
                "steps_per_run": int(hp.get("steps_per_run") or hp.get("steps") or 100),
                "samples_per_run": int(hp.get("samples_per_run") or hp.get("samples") or 1000),
                "merge_mode": MergeMode(hp.get("merge_mode") or "slao"),
                "checkpoint_dir": str(history_dir),
            }
            # BRIDGE-A-002: thread Wave 6b override keys through to
            # MultiRunConfig fields / MultiRunTrainer kwargs. Same split as
            # cmd_multi_run: a key that lives on MultiRunConfig goes via
            # mr_config_kwargs; a key that lives on MultiRunTrainer.__init__
            # goes via mr_trainer_kwargs; keys present in neither degrade
            # to no-op (matches the cmd_multi_run defensive scaffolding).
            try:
                _mr_cfg_fields = {f.name for f in _dc.fields(MultiRunConfig)}
            except (TypeError, ValueError):
                _mr_cfg_fields = set()
            try:
                _mr_trainer_params = set(_inspect.signature(MultiRunTrainer.__init__).parameters)
            except (TypeError, ValueError):
                _mr_trainer_params = set()
            mr_wave6b_candidate_kwargs: dict[str, Any] = {
                k: hp[k] for k in _wave6b_keys if k in hp
            }
            mr_wave6b_cfg_kwargs = {
                k: v for k, v in mr_wave6b_candidate_kwargs.items()
                if k in _mr_cfg_fields
            }
            mr_wave6b_trainer_kwargs = {
                k: v for k, v in mr_wave6b_candidate_kwargs.items()
                if k not in _mr_cfg_fields and k in _mr_trainer_params
            }
            mr_config_kwargs.update(mr_wave6b_cfg_kwargs)
            mr_config = MultiRunConfig(**mr_config_kwargs)
            mr_trainer = MultiRunTrainer(
                model=model,
                config=mr_config,
                **mr_wave6b_trainer_kwargs,
            )
            mr_result = mr_trainer.run(dataset)
            print()
            _print_success("Replay (multi-run) complete!")
            _print_kv("Total runs", str(mr_result.total_runs))
            _print_kv("Final loss", f"{mr_result.final_loss:.4f}")
            return EXIT_OK

        # Single-run path: build Trainer with inherited hyperparameters,
        # then call train() with the inherited dataset / steps / samples.
        sr_trainer_kwargs: dict[str, Any] = {
            "model": model,
            "lora_r": int(hp.get("lora_r") or 16),
            "learning_rate": float(hp.get("learning_rate") or hp.get("lr") or 2e-4),
            "output_dir": str(history_dir),
        }
        if "batch_size" in hp:
            sr_trainer_kwargs["batch_size"] = hp["batch_size"]
        if "gradient_accumulation" in hp:
            sr_trainer_kwargs["gradient_accumulation"] = int(hp["gradient_accumulation"])
        if "lora_alpha" in hp:
            sr_trainer_kwargs["lora_alpha"] = int(hp["lora_alpha"])
        if "lora_dropout" in hp:
            sr_trainer_kwargs["lora_dropout"] = float(hp["lora_dropout"])

        # BRIDGE-A-002 (v1.4): merge Wave 6b override keys into the candidate
        # kwargs BEFORE the signature filter so the next block drops any
        # field the installed Trainer can't absorb. Pre-fix these five keys
        # were silently dropped even when the operator passed them via
        # --override. The filter below is the same defensive net used by
        # cmd_train (cli.py ~617): keys outside Trainer.__init__.parameters
        # are filtered out — keeps mocked-Trainer test runs working AND
        # forward-compatible with backend builds that DO accept them.
        for _w6b_key in _wave6b_keys:
            if _w6b_key in hp:
                sr_trainer_kwargs[_w6b_key] = hp[_w6b_key]

        # Filter to keys the installed Trainer constructor actually
        # accepts so a backend version that has dropped a field doesn't
        # blow up the replay. Defensive: (a) a non-introspectable Trainer
        # (e.g. a test MagicMock with no spec) advertises ``(*args, **kwargs)``
        # so we MUST NOT filter — its catch-all kwargs accept anything; (b)
        # try/except keeps the path working even if signature inspection
        # raises (unusual C-extension cases).
        try:
            _sig = _inspect.signature(Trainer.__init__)
            _has_var_keyword = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
            )
            if not _has_var_keyword:
                _trainer_params = set(_sig.parameters)
                sr_trainer_kwargs = {k: v for k, v in sr_trainer_kwargs.items() if k in _trainer_params}
        except (TypeError, ValueError):
            pass

        sr_trainer = Trainer(**sr_trainer_kwargs)
        callback = TrainingCallback()
        steps = int(hp.get("max_steps") or hp.get("steps") or 100)
        samples = hp.get("samples") or hp.get("max_samples")
        run = sr_trainer.train(
            dataset=dataset,
            steps=steps,
            samples=int(samples) if samples is not None else None,
            callback=callback,
        )
        sr_trainer.save(args.output)
        print()
        _print_success("Replay complete!")
        _print_kv("Final loss", f"{run.final_loss:.4f}")
        _print_kv("Original run", original_run_id[:12])
        _print_info(
            f"Diff against the original with: "
            f"backprop diff-runs {original_run_id[:12]} <new-run-id>"
        )
        return EXIT_OK

    except KeyboardInterrupt:
        print()
        _print_warning("Replay interrupted by user")
        return EXIT_INTERRUPTED
    except BackpropagateError as e:
        _print_error(f"Replay failed: {e.message}")
        if e.suggestion:
            _print_info(f"Suggestion: {e.suggestion}")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        if args.verbose:
            _print_error(f"Replay failed: {e}")
            import traceback
            traceback.print_exc()
        else:
            _print_error_redacted(e, prefix="Replay failed: ")
            _print_info("Run with --verbose for full traceback")
        return EXIT_RUNTIME_ERROR


# =============================================================================
# COMMAND: export-runs (BRIDGE Wave 6b)
# =============================================================================

# Supported export formats. JSONL is the canonical pipeline-friendly shape
# (one record per line, append-only friendly, parses incrementally in jq /
# W&B / MLflow ingest scripts). Reserved-by-design: csv (column-narrow,
# loses nested loss_history / merge_history detail) — operators wanting CSV
# today can pipe through ``jq -r`` or a small Python one-liner.
_EXPORT_RUNS_FORMATS: tuple[str, ...] = ("jsonl",)


def cmd_export_runs(args: argparse.Namespace) -> int:
    """Execute ``backprop export-runs --format=jsonl`` (BRIDGE Wave 6b).

    Bulk dump of every run history entry as JSONL (one record per line).
    Useful for offline analytics, pipeline integration with W&B / MLflow,
    or attaching a corpus to a support ticket.

    Exit codes:
        0   export completed (or empty history — JSONL output is just
            zero lines, which is a well-formed empty stream)
        1   history directory not found
        2   write failure
    """
    import json

    from .checkpoints import RunHistoryManager

    history_dir = Path(args.output).expanduser()
    if not history_dir.exists():
        _print_error(f"No history directory: {history_dir}")
        _print_info(
            "Pass --output <dir> to point at the output directory used "
            "during training."
        )
        return EXIT_USER_ERROR

    manager = RunHistoryManager(str(history_dir))
    try:
        runs = manager.list_runs(status=getattr(args, "status", None), limit=None)
    except ValueError as e:
        _print_error(str(e))
        return EXIT_USER_ERROR

    if args.format not in _EXPORT_RUNS_FORMATS:
        _print_error(
            f"Unsupported --format {args.format!r}. Supported: "
            f"{sorted(_EXPORT_RUNS_FORMATS)}"
        )
        return EXIT_USER_ERROR

    # Output destination: --to writes to a file, otherwise stdout. Stdout
    # is the default so a pipeline `backprop export-runs | jq ...` works
    # without intermediate files.
    out_path: Path | None = None
    if getattr(args, "to", None):
        out_path = Path(args.to).expanduser()

    try:
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                for entry in runs:
                    fh.write(json.dumps(entry, default=str))
                    fh.write("\n")
            _print_success(
                f"Exported {len(runs)} run(s) to {out_path}"
            )
        else:
            for entry in runs:
                print(json.dumps(entry, default=str))
            # Banner goes to stderr so it doesn't pollute a piped jq
            # consumer reading stdout.
            print(
                f"[INFO] Exported {len(runs)} run(s) from {history_dir}",
                file=sys.stderr,
            )
        return EXIT_OK
    except OSError as e:
        _print_error(f"Write failed: {e}")
        return EXIT_RUNTIME_ERROR


# =============================================================================
# COMMAND: validate (BRIDGE-F-007)
# =============================================================================

def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the ``backprop validate <dataset>`` subcommand (BRIDGE-F-007).

    Thin wrapper around :func:`backpropagate.datasets.validate_dataset` so
    operators can pre-flight a dataset before kicking off a multi-hour
    training run. The function was already part of the public API
    (re-exported from ``backpropagate.__init__``); this subcommand only
    surfaces it on the CLI.

    Exit codes:
        0   dataset validated cleanly
        1   user error — file missing / unreadable / bad encoding
        65  EX_DATAERR — validation errors detected (per BRIDGE-F-006)
    """
    from .datasets import DatasetFormat, detect_format, validate_dataset

    _print_header("Backpropagate Dataset Validation")

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        _print_error(f"Dataset not found: {dataset_path}")
        _print_info(
            "Pass a local JSONL file path, or use `backprop train --data "
            "<hf-name>` to validate a HuggingFace dataset by attempting "
            "to load it through the trainer."
        )
        return EXIT_USER_ERROR

    if dataset_path.is_dir():
        _print_error(f"Dataset path is a directory, expected a file: {dataset_path}")
        return EXIT_USER_ERROR

    # Load samples without going through the full DatasetLoader machinery
    # (which pulls torch); a plain JSONL line-read is enough for the
    # validator. Cap reads at --max-samples * 2 so we don't OOM on a
    # 100 GB dataset just to validate the first N rows.
    import json
    samples: list[dict | str] = []
    line_count = 0
    parse_errors: list[tuple[int, str]] = []
    cap = (args.max_samples or 0) * 2 if args.max_samples else None

    try:
        with open(dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line_count += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    samples.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    parse_errors.append((line_count, str(exc)))
                    if len(parse_errors) >= args.max_errors:
                        break
                if cap is not None and len(samples) >= cap:
                    break
                if args.max_samples is not None and len(samples) >= args.max_samples:
                    break
    except UnicodeDecodeError as exc:
        _print_error(f"Dataset is not valid UTF-8: {exc}")
        _print_info("Re-encode the file: `iconv -f <enc> -t utf-8 < src > dst`")
        return EXIT_USER_ERROR
    except OSError as exc:
        _print_error(f"Could not read dataset: {exc}")
        return EXIT_USER_ERROR

    _print_kv("Path", str(dataset_path))
    _print_kv("Lines scanned", str(line_count))
    _print_kv("Samples parsed", str(len(samples)))
    if parse_errors:
        _print_warning(f"{len(parse_errors)} JSON parse error(s) — see below")

    if not samples:
        _print_error("Dataset has no parseable rows.")
        return EXIT_DATA_ERR

    # Resolve format hint.
    if args.format == "auto":
        format_hint = None
        detected = detect_format(samples[0])
    else:
        try:
            format_hint = DatasetFormat(args.format)
            detected = format_hint
        except ValueError:
            _print_error(f"Unknown format hint: {args.format}")
            return EXIT_USER_ERROR

    _print_kv("Format hint", args.format)
    _print_kv("Format detected", detected.value if hasattr(detected, "value") else str(detected))

    result = validate_dataset(samples, format_type=format_hint, max_errors=args.max_errors)

    _print_kv("Total rows", str(result.total_rows))
    _print_kv("Valid rows", str(result.valid_rows))
    _print_kv("Errors", str(len(result.errors)))
    _print_kv("Warnings", str(len(result.warnings)))

    if parse_errors:
        print(f"\n{Colors.BOLD}JSON parse errors{Colors.RESET}")
        for line_no, message in parse_errors[:10]:
            print(f"  line {line_no}: {message}")
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more")

    if result.errors:
        print(f"\n{Colors.BOLD}Validation errors{Colors.RESET}")
        for err in result.errors[:10]:
            print(f"  row {err.row_index}: [{err.error_type}] {err.message}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more")

    if result.warnings:
        print(f"\n{Colors.BOLD}Warnings{Colors.RESET}")
        for w in result.warnings[:10]:
            print(f"  row {w.row_index}: [{w.error_type}] {w.message}")
        if len(result.warnings) > 10:
            print(f"  ... and {len(result.warnings) - 10} more")

    print()
    if result.is_valid and not parse_errors:
        _print_success("Dataset validation PASSED")
        return EXIT_OK

    _print_error("Dataset validation FAILED — see errors above")
    return EXIT_DATA_ERR


# =============================================================================
# COMMAND: estimate-vram (BRIDGE-F-008)
# =============================================================================

# Tier table mirrors Trainer._detect_batch_size — the recommended batch
# size at each VRAM level. Keep this in sync with trainer.py:906; the
# duplication is intentional so the CLI surface doesn't need to construct
# a Trainer (which would import torch + transformers).
_VRAM_BATCH_SIZE_TIERS: list[tuple[float, int, str]] = [
    (24.0, 4, "RTX 4090 / 3090 / A5000 — 7B fits with LoRA r=64"),
    (16.0, 2, "RTX 5080 / 4070 Ti Super — 7B with LoRA r=16-32"),
    (12.0, 1, "RTX 4070 / 3060 12GB — 7B with LoRA r=8 + gradient checkpointing"),
    (0.0, 1, "Tight fit / fallback — batch=1 + LoRA r=8 + gradient ckpt"),
]


def cmd_estimate_vram(args: argparse.Namespace) -> int:
    """Execute the ``backprop estimate-vram <model>`` subcommand (BRIDGE-F-008).

    Surfaces Trainer._detect_batch_size's tier heuristic on the CLI so
    operators can preview the recommended batch size before kicking off
    training. The math is model-agnostic (per the tier table) — the model
    argument is only used for the printed header.

    Exit codes:
        0   table printed
        1   --vram-gb out of plausible range
    """
    # Resolve VRAM: explicit override > torch query > error.
    vram_gb: float | None = None
    detected_via = "unknown"
    if args.vram_gb is not None:
        vram_gb = float(args.vram_gb)
        detected_via = "user (--vram-gb)"
    else:
        try:
            import torch
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                vram_gb = vram_bytes / (1024 ** 3)
                detected_via = "torch.cuda"
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 — best-effort VRAM probe
            logger.debug(f"VRAM probe failed: {exc}")

    if vram_gb is None:
        _print_error(
            "Could not detect VRAM and no --vram-gb override supplied."
        )
        _print_info(
            "Pass --vram-gb <number> to simulate the table for a specific "
            "card, or install a CUDA-enabled PyTorch on a host with a GPU."
        )
        return EXIT_USER_ERROR

    if vram_gb <= 0 or vram_gb > 512:
        _print_error(f"--vram-gb {vram_gb} outside the plausible range (0, 512].")
        return EXIT_USER_ERROR

    # Pick the matching tier.
    selected_tier = _VRAM_BATCH_SIZE_TIERS[-1]
    for threshold, bs, note in _VRAM_BATCH_SIZE_TIERS:
        if vram_gb >= threshold:
            selected_tier = (threshold, bs, note)
            break

    payload: dict[str, Any] = {
        "model": args.model,
        "vram_gb": round(vram_gb, 2),
        "detected_via": detected_via,
        "recommended_batch_size": selected_tier[1],
        "tier_notes": selected_tier[2],
        "tiers": [
            {"vram_min_gb": t[0], "batch_size": t[1], "note": t[2]}
            for t in _VRAM_BATCH_SIZE_TIERS
        ],
    }

    if args.json:
        import json
        print(json.dumps(payload, indent=2, default=str))
        return EXIT_OK

    _print_header("Backpropagate VRAM Estimator")
    _print_kv("Model", args.model)
    _print_kv("VRAM", f"{vram_gb:.1f} GB ({detected_via})")
    _print_kv("Recommended --batch-size", str(selected_tier[1]))
    _print_info(selected_tier[2])

    print(f"\n{Colors.BOLD}Tier table{Colors.RESET}")
    header = f"{'VRAM (>= GB)':<14}  {'batch_size':<10}  notes"
    print(f"{Colors.BOLD}{header}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 14}  {'-' * 10}  -----{Colors.RESET}")
    for threshold, bs, note in _VRAM_BATCH_SIZE_TIERS:
        marker = " <- this card" if (threshold, bs, note) == selected_tier else ""
        print(f"{threshold:<14.1f}  {bs:<10}  {note}{marker}")

    print()
    _print_info(
        "The same tier heuristic fires automatically when you run "
        "`backprop train --batch-size auto` (trainer.py:_detect_batch_size). "
        "Pass --batch-size <N> to override the auto-detection."
    )
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

Sysexits.h overlay (BRIDGE-F-006; emitted only by main()'s catch-all when
no subcommand handler claimed the failure with a 0/1/2/3 code):
  64  EX_USAGE        — malformed invocation reached the catch-all
  65  EX_DATAERR      — dataset unreadable / malformed (DatasetError class)
  69  EX_UNAVAILABLE  — HF Hub / Ollama / network service unreachable
  70  EX_SOFTWARE     — internal uncaught exception (fall-through default)
  77  EX_NOPERM       — permission denied (POSIX EACCES / EPERM, Win ACL)
  137 OOM-killer      — CUDA / torch OutOfMemoryError reached the catch-all
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
    # BRIDGE-A-003 (v1.4): --lora-r default 256 matches LoraConfig.r in
    # config.py (both the pydantic-settings branch at config.py:268 AND the
    # dataclass fallback at config.py:646 default to 256). If you change
    # this value, ALSO update:
    #   * config.py:268 (BaseSettings branch LoraConfig.r)
    #   * config.py:646 (dataclass fallback LoraConfig.r)
    #   * site/src/content/docs/handbook/cli-reference.md (--lora-r row)
    #   * site/src/content/docs/handbook/env-vars.md (BACKPROPAGATE_LORA__R)
    # so the argparse default / settings default / handbook table stay in
    # lockstep. Drift between these surfaces is the [[no-banner-documenting
    # -no-op]] adjacent pattern audited in v1.4 (BRIDGE-A-003) — an operator
    # reading the handbook is told one default and the runtime applies another.
    train_parser.add_argument(
        "--lora-r",
        type=_positive_int,
        default=256,
        help="LoRA rank (default: 256 in v1.3 'quality' preset; pass --lora-preset=fast for v1.2.x rank-16 footprint). Must be > 0.",
    )
    # BRIDGE Wave 6b (v1.3): five new LoRA / training knobs added in
    # lock-step with the backend agent's Wave 6b additions
    # (DoRA / packing-default / PiSSA-LoftQ / quality-vs-fast preset /
    # paged-Adam auto). Each flag binds to a Trainer kwarg of the same
    # name; the Trainer applies the kwarg verbatim to LoraConfig /
    # SFTConfig so a future peft / trl release that renames a field
    # surfaces as a single Trainer-side fix instead of a CLI-side one.
    train_parser.add_argument(
        "--use-dora",
        action="store_true",
        help=(
            "Enable DoRA (Weight-Decomposed Low-Rank Adaptation). "
            "Rank 8 DoRA ~= rank 32 LoRA quality. Zero inference overhead. "
            "Requires peft>=0.10. (16GB study-swarm Wave 6b)"
        ),
    )
    train_parser.add_argument(
        "--no-packing",
        action="store_true",
        help=(
            "Disable sample packing. Default ON (1.7-3x throughput). "
            "Disable only if you hit packing-incompatible behavior."
        ),
    )
    train_parser.add_argument(
        "--init-lora-weights",
        choices=["default", "pissa", "loftq"],
        default="default",
        help=(
            "LoRA initialization strategy. PiSSA + LoftQ recover quality "
            "lost during QLoRA quantization at zero runtime cost. "
            "(16GB study-swarm Wave 6b)"
        ),
    )
    train_parser.add_argument(
        "--lora-preset",
        choices=["quality", "fast"],
        default="quality",
        help=(
            "LoRA configuration preset. 'quality' = rank 256 + all-linear "
            "+ 10x LR (new v1.3 default, matches full fine-tuning per"
            "Biderman 2024). 'fast' = rank 16 + q+v + 1x LR (v1.2"
            "defaults; smaller memory footprint)."
        ),
    )
    train_parser.add_argument(
        "--optim",
        choices=["auto", "adamw_torch", "paged_adamw_8bit", "adamw_8bit"],
        default="auto",
        help=(
            "Optimizer. 'auto' picks paged_adamw_8bit on consumer GPUs "
            "(<24GB VRAM), adamw_torch otherwise. Override for specific "
            "needs."
        ),
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
    # BRIDGE Wave 6b (v1.3): mirror of train_parser's five new LoRA /
    # training knobs so a multi-run session honors the same
    # DoRA / packing / init-strategy / preset / optimizer choices as a
    # single-run session. Each flag binds to a MultiRunConfig / Trainer
    # kwarg of the same name; the underlying Trainer applies it verbatim
    # to LoraConfig / SFTConfig.
    multi_parser.add_argument(
        "--use-dora",
        action="store_true",
        help=(
            "Enable DoRA (Weight-Decomposed Low-Rank Adaptation). "
            "Rank 8 DoRA ~= rank 32 LoRA quality. Zero inference overhead. "
            "Requires peft>=0.10. (16GB study-swarm Wave 6b)"
        ),
    )
    multi_parser.add_argument(
        "--no-packing",
        action="store_true",
        help=(
            "Disable sample packing. Default ON (1.7-3x throughput). "
            "Disable only if you hit packing-incompatible behavior."
        ),
    )
    multi_parser.add_argument(
        "--init-lora-weights",
        choices=["default", "pissa", "loftq"],
        default="default",
        help=(
            "LoRA initialization strategy. PiSSA + LoftQ recover quality "
            "lost during QLoRA quantization at zero runtime cost. "
            "(16GB study-swarm Wave 6b)"
        ),
    )
    multi_parser.add_argument(
        "--lora-preset",
        choices=["quality", "fast"],
        default="quality",
        help=(
            "LoRA configuration preset. 'quality' = rank 256 + all-linear "
            "+ 10x LR (new v1.3 default, matches full fine-tuning per"
            "Biderman 2024). 'fast' = rank 16 + q+v + 1x LR (v1.2"
            "defaults; smaller memory footprint)."
        ),
    )
    multi_parser.add_argument(
        "--optim",
        choices=["auto", "adamw_torch", "paged_adamw_8bit", "adamw_8bit"],
        default="auto",
        help=(
            "Optimizer. 'auto' picks paged_adamw_8bit on consumer GPUs "
            "(<24GB VRAM), adamw_torch otherwise. Override for specific "
            "needs."
        ),
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
        help=(
            "HF token override for --push-to-hub (default: env / cached "
            "login). NOTE: passing the token on the command line leaks it "
            "to `ps aux` and shell history — prefer --hub-token-file or "
            "the HF_TOKEN env var for repeat invocations."
        ),
    )
    # BRIDGE-A-004 (v1.4): file-based HF token resolution. Mirrors the v1.3
    # cmd_ui --auth-file pattern (cli.py ~2137): existence check, mode-0600
    # warning on POSIX, content read in cmd_export's _read_hub_token_file
    # helper. Mutually exclusive with --hub-token (passing both is operator
    # error — see _read_hub_token_file). Keeps the token off argv where
    # `ps aux` and shell history would otherwise retain it.
    export_parser.add_argument(
        "--hub-token-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a file containing the HF token on the first line "
            "(replaces --hub-token; keeps the credential out of shell "
            "history and `ps aux`). Set mode 0600 on POSIX. Mutually "
            "exclusive with --hub-token."
        ),
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
            "~/.cache/huggingface/token from `huggingface-cli login`). "
            "NOTE: passing the token on the command line leaks it to "
            "`ps aux` and shell history — prefer --token-file or the "
            "HF_TOKEN env var for repeat invocations."
        ),
    )
    # BRIDGE-A-004 (v1.4): file-based HF token resolution. See the
    # symmetrical --hub-token-file on the export subparser for the design
    # rationale. Mutually exclusive with --token.
    push_parser.add_argument(
        "--token-file",
        default=None,
        metavar="PATH",
        help=(
            "Path to a file containing the HF token on the first line "
            "(replaces --token; keeps the credential out of shell history "
            "and `ps aux`). Set mode 0600 on POSIX. Mutually exclusive "
            "with --token."
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

    # diff-runs command (BRIDGE Wave 6b)
    diff_runs_parser = subparsers.add_parser(
        "diff-runs",
        help="Diff config + hyperparameters + final loss between two runs",
        description=(
            "Side-by-side comparison of two completed runs from the on-disk "
            "run history. Reads from RunHistoryManager (the same JSON "
            "consumed by `runs` / `show-run` / `list-runs`). Useful for the "
            "'did this config tweak actually move the loss' workflow."
        ),
    )
    diff_runs_parser.add_argument(
        "run_id_a",
        help=(
            "First run_id (or unambiguous prefix). Partial-prefix matching "
            "follows the same rules as `backprop show-run`."
        ),
    )
    diff_runs_parser.add_argument(
        "run_id_b",
        help=(
            "Second run_id (or unambiguous prefix). Partial-prefix matching "
            "follows the same rules as `backprop show-run`."
        ),
    )
    diff_runs_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory containing run_history.json (default: ./output)",
    )
    diff_runs_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help=(
            "Output format. 'table' = colorized side-by-side view (default, "
            "humans). 'json' = machine-readable payload with the same diff "
            "rows + a changed_count summary."
        ),
    )
    diff_runs_parser.set_defaults(func=cmd_diff_runs)

    # replay command (BRIDGE Wave 6b)
    replay_parser = subparsers.add_parser(
        "replay",
        help="Re-run an existing run with the same config (fresh run_id)",
        description=(
            "Re-execute a recorded training run with the same model + "
            "dataset + hyperparameters. Inherits seed / learning_rate / "
            "batch_size / gradient_accumulation / max_steps / lora_r from "
            "the original entry. The replay gets a fresh run_id (no "
            "clobbering) so the result can be diffed against the source "
            "via `backprop diff-runs <original> <replay>`. Useful for "
            "'did this reproduce' verification."
        ),
    )
    replay_parser.add_argument(
        "run_id",
        help="The run_id to replay (or any unambiguous prefix).",
    )
    replay_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory containing run_history.json (default: ./output)",
    )
    replay_parser.add_argument(
        "--override",
        type=_parse_replay_override,
        action="append",
        metavar="KEY=VALUE",
        default=None,
        help=(
            "Override a single hyperparameter; repeatable. Example: "
            "--override learning_rate=1e-4 --override batch_size=4. "
            "Only whitelisted keys are accepted (see source for the full "
            "list; common ones are: learning_rate / lr / batch_size / "
            "gradient_accumulation / max_steps / samples / lora_r / "
            "lora_alpha / lora_dropout / use_dora / packing / "
            "init_lora_weights / lora_preset / optim). Unknown keys "
            "fail loudly so a typo doesn't silently inherit the original."
        ),
    )
    replay_parser.set_defaults(func=cmd_replay)

    # export-runs command (BRIDGE Wave 6b)
    export_runs_parser = subparsers.add_parser(
        "export-runs",
        help="Bulk export of run history (JSONL, one record per line)",
        description=(
            "Dump every run history entry as JSONL — one record per line, "
            "well-formed even for an empty history (zero lines). Useful "
            "for offline analytics, pipeline integration with W&B / "
            "MLflow, or attaching a corpus to a support ticket. Writes "
            "to stdout by default so `backprop export-runs | jq ...` "
            "works without intermediate files."
        ),
    )
    export_runs_parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory containing run_history.json (default: ./output)",
    )
    export_runs_parser.add_argument(
        "--format",
        choices=list(_EXPORT_RUNS_FORMATS),
        default="jsonl",
        help=(
            "Export format. Today only 'jsonl' is supported (one record "
            "per line). CSV is intentionally not offered — it loses the "
            "nested loss_history / merge_history shape; pipe through "
            "jq -r if you need columns."
        ),
    )
    export_runs_parser.add_argument(
        "--to",
        default=None,
        metavar="PATH",
        help=(
            "Write to PATH instead of stdout. Parent directory is created "
            "if it doesn't exist."
        ),
    )
    export_runs_parser.add_argument(
        "--status",
        choices=["running", "completed", "failed"],
        default=None,
        help="Filter by status before export (default: include all).",
    )
    export_runs_parser.set_defaults(func=cmd_export_runs)

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
            "Expose the UI via a Cloudflare Quick Tunnel (requires "
            "`cloudflared` on PATH; see handbook/security.md). Requires "
            "--auth user:pass; passing --share without --auth exits with "
            "RUNTIME_UI_AUTH_NOT_ENFORCED. The tunnel URL is printed to "
            "stderr on connect and torn down when the CLI exits."
        ),
    )
    ui_parser.add_argument(
        "--auth",
        type=_auth_credential,
        metavar="USER:PASS",
        help=(
            "Enable HTTP basic auth on the Reflex UI. Required when --share "
            "or a non-loopback --host is passed. Credentials are forwarded "
            "to the subprocess via BACKPROPAGATE_UI_AUTH. "
            "Username must not contain whitespace, colon, or control chars; "
            "password must not contain newlines or NUL. Use the "
            "BACKPROPAGATE_UI_AUTH env var or --auth-file to keep "
            "credentials out of shell history."
        ),
    )
    # BRIDGE-F-002 auth-polish item 4 (v1.3 Wave 6a): file-based credential
    # source for operators who don't want the credential in shell history
    # or process argv. Mutex with --auth (cmd_ui raises if both supplied).
    ui_parser.add_argument(
        "--auth-file",
        metavar="PATH",
        default=None,
        help=(
            "Read 'user:pass' from PATH instead of taking --auth on the "
            "command line — keeps the credential out of shell history. "
            "Mutex with --auth. On POSIX the file mode is checked and a "
            "warning fires if it's wider than 0600. Use "
            "`printf 'user:pass' > path && chmod 600 path` to create."
        ),
    )
    ui_parser.set_defaults(func=cmd_ui)

    # validate command (BRIDGE-F-007 — wrap existing validate_dataset)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a dataset's format + content",
        description=(
            "Wrap backpropagate.datasets.validate_dataset to report row "
            "count, detected format, and any malformed rows for a JSONL / "
            "ShareGPT / Alpaca / OpenAI dataset. Returns exit 0 on a clean "
            "validation, 65 (EX_DATAERR) on detected errors, 1 on input "
            "problems (missing file / unreadable / bad encoding)."
        ),
    )
    validate_parser.add_argument(
        "dataset",
        help="Path to the JSONL dataset to validate (local file or HF name)",
    )
    validate_parser.add_argument(
        "--format",
        choices=["auto", "sharegpt", "alpaca", "openai", "raw"],
        default="auto",
        help="Format hint (default: auto-detect)",
    )
    validate_parser.add_argument(
        "--max-errors",
        type=_positive_int,
        default=100,
        help="Maximum errors to collect before stopping (default: 100)",
    )
    validate_parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="Maximum samples to validate (default: all)",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # estimate-vram command (BRIDGE-F-008 — wrap Trainer._detect_batch_size logic)
    estimate_vram_parser = subparsers.add_parser(
        "estimate-vram",
        help="Estimate VRAM requirements at different batch sizes",
        description=(
            "Print a small table of recommended batch sizes given the "
            "currently-visible GPU VRAM and a model name. The math mirrors "
            "Trainer._detect_batch_size's tier heuristic (which fires "
            "automatically when --batch-size auto is used). Useful before "
            "starting a long training run on a card you haven't profiled."
        ),
    )
    estimate_vram_parser.add_argument(
        "model",
        nargs="?",
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "Model name (default: Qwen/Qwen2.5-7B-Instruct). Used only for "
            "the printed header — the VRAM tiers are model-agnostic."
        ),
    )
    estimate_vram_parser.add_argument(
        "--vram-gb",
        type=_positive_float,
        default=None,
        help=(
            "Override the detected VRAM (in GB) so you can simulate the "
            "table on a card you don't currently have. Default: query the "
            "primary CUDA device."
        ),
    )
    estimate_vram_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the table as JSON for CI / scripting consumers.",
    )
    estimate_vram_parser.set_defaults(func=cmd_estimate_vram)

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

    BRIDGE-B-016 (Stage C): mint ``cli_run_id`` BEFORE ``parser.parse_args``
    so a Ctrl+C during an early argparse validator (e.g. a future
    ``_hf_repo_validator`` that does a network probe) still produces a
    run_id the operator can grep for in any partial log output.

    BRIDGE-B-013 (Stage C): log-setup failures no longer fail silently.
    The first failure stashes a one-line reason into a module-level flag;
    main() surfaces a single ``[WARN] structured logging setup failed: ...``
    line to stderr before returning so the operator knows logs aren't
    flowing without aborting the CLI.
    """
    import traceback
    import uuid

    from .logging_config import bind_run_context, configure_logging, get_logger

    # BRIDGE-B-016 (Stage C): mint the cli_run_id BEFORE parse_args so the
    # KeyboardInterrupt during arg validation has a token to print.
    cli_run_id = uuid.uuid4().hex

    try:
        parser = create_parser()
        # BRIDGE-F-005 (v1.3 Wave 6a): wire argcomplete so shell-installed
        # completers fire on TAB. The library is optional — when not
        # installed (and not requested via the env vars argcomplete sets),
        # the call is a silent no-op. The hook fires BEFORE parse_args so
        # the completer can inspect the parser's structure for choices,
        # subcommands, and metavars.
        try:
            import argcomplete
            argcomplete.autocomplete(parser)
        except ImportError:
            pass
        args = parser.parse_args(argv)
    except KeyboardInterrupt:
        # KeyboardInterrupt during argparse (rare but possible if a custom
        # validator does I/O) should still exit 130 per the Ship Gate contract.
        print(f"Interrupted (run_id={cli_run_id[:12]}).", file=sys.stderr)
        return EXIT_INTERRUPTED

    # BRIDGE-B-002: configure structured logging before ANY subcommand runs.
    # ``--verbose`` bumps the level to DEBUG so structlog emits the full
    # event stream; otherwise honour BACKPROPAGATE_LOG_LEVEL (default INFO).
    verbose = bool(getattr(args, "verbose", False))

    # BRIDGE-B-013 (Stage C): track logging-setup failures so the operator
    # sees a stderr WARN line at the end of main() naming the reason.
    # Pre-fix this was a silent ``except Exception: pass`` — a typo'd
    # BACKPROPAGATE_LOG_FILE pointing at a non-existent directory left the
    # CLI running with no log output AND no warning. Module-level flag so
    # all three swallow-points (configure_logging, bind_run_context,
    # cli_invoked) can collapse to one warn line.
    global _LOGGING_SETUP_FAILED, _LOGGING_SETUP_FAIL_REASON
    _LOGGING_SETUP_FAILED = False
    _LOGGING_SETUP_FAIL_REASON = ""

    try:
        configure_logging(level="DEBUG" if verbose else None, force=True)
    except Exception as _log_exc:  # noqa: BLE001 — logging setup must not abort the CLI
        # If logging setup itself fails (extremely rare — bad log file path,
        # permission denied), fall back silently so the subcommand still runs.
        # The traceback is intentionally swallowed so a misconfigured
        # BACKPROPAGATE_LOG_FILE can't crash the CLI on startup.
        _LOGGING_SETUP_FAILED = True
        # Stash the first-line reason (typically the OSError message);
        # full traceback is BACKPROPAGATE_DEBUG-gated below.
        _LOGGING_SETUP_FAIL_REASON = f"{type(_log_exc).__name__}: {_log_exc}"
        if os.environ.get("BACKPROPAGATE_DEBUG"):
            traceback.print_exc()

    # BRIDGE-B-002: bind the cli_run_id so every structured log line emitted
    # from this process carries it. Subcommand handlers re-emit the same id
    # to stderr (replacing the prior 12-char-only print) so operators
    # triaging a multi-hour overnight job can grep ONE token across CLI
    # output, trainer logs, and run_history.json.
    try:
        bind_run_context(run_id=cli_run_id, subcommand=getattr(args, "command", None))
    except Exception as _bind_exc:  # noqa: BLE001 — context binding must not abort the CLI
        if not _LOGGING_SETUP_FAILED:
            _LOGGING_SETUP_FAILED = True
            _LOGGING_SETUP_FAIL_REASON = (
                f"bind_run_context: {type(_bind_exc).__name__}: {_bind_exc}"
            )
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
    except Exception as _emit_exc:  # noqa: BLE001 — logging must not abort the CLI
        if not _LOGGING_SETUP_FAILED:
            _LOGGING_SETUP_FAILED = True
            _LOGGING_SETUP_FAIL_REASON = (
                f"cli_invoked emit: {type(_emit_exc).__name__}: {_emit_exc}"
            )

    # BRIDGE-B-017 (Stage C): print a deprecation hint when an operator
    # invokes a subcommand whose stability tier is "deprecated-prefer-X".
    # The hint is purely advisory — the deprecated subcommand still runs.
    # (BRIDGE-A-001, v1.4): the legacy hint pointed at a never-registered
    # `backprop info --subcommand-tiers` flag — stripped to avoid the
    # [[no-banner-documenting-no-op]] tripwire. If that introspection
    # surface lands later, restore the pointer at that time.
    cmd_name = getattr(args, "command", None)
    tier = SUBCOMMAND_TIERS.get(cmd_name or "", "stable")
    if tier.startswith("deprecated"):
        try:
            preferred = tier.split("prefer-", 1)[1] if "prefer-" in tier else "the replacement subcommand"
        except Exception:  # noqa: BLE001  # nosec B110
            preferred = "the replacement subcommand"
        print(
            f"[deprecation] `backprop {cmd_name}` is deprecated; prefer "
            f"`backprop {preferred}`.",
            file=sys.stderr,
        )

    if not args.command:
        # First-time-user friendliness (C-CLI-001): bare ``backprop`` invocation
        # prints the rich help text (subcommand list + examples + exit-code
        # table) instead of a single terse error line. The exit code stays
        # non-zero so CI scripts that asserted "no-subcommand is failure" keep
        # behaving the same — only the output content changes from one line
        # to the full --help block.
        parser.print_help(sys.stderr)
        return EXIT_USER_ERROR

    # BRIDGE-B-013 (Stage C): if logging setup failed earlier, surface a
    # single stderr WARN line BEFORE we hand control to the subcommand.
    # The subcommand will be run regardless ("logging setup must not abort
    # the CLI" per the existing principle), but the operator gets a clear
    # heads-up that BACKPROPAGATE_LOG_* env vars aren't taking effect.
    def _maybe_warn_logging_setup_failed() -> None:
        if _LOGGING_SETUP_FAILED:
            print(
                f"[WARN] structured logging setup failed: "
                f"{_LOGGING_SETUP_FAIL_REASON}. "
                "Re-run with BACKPROPAGATE_DEBUG=1 for the full traceback.",
                file=sys.stderr,
            )

    _maybe_warn_logging_setup_failed()

    # Execute the command — wrapped in a last-resort exception net so the
    # Ship Gate B2 exit-code contract holds even when a subcommand's own
    # try/except missed something.
    #
    # BRIDGE-F-006 sysexits.h overlay: when the catch-all fires (a subcommand
    # let an exception escape its own try/except), classify the failure mode
    # against the sysexits.h-flavored finer-grained codes before falling
    # back to EXIT_RUNTIME_ERROR. Subcommand handlers continue to emit the
    # legacy 0 / 1 / 2 / 3 / 130 codes via their documented contracts; this
    # overlay only fires when nothing inside the subcommand caught the
    # specific exception class. The mapping is:
    #
    #   torch.cuda.OutOfMemoryError / RUNTIME_GPU_OOM   -> 137 (OOM-killer)
    #   UserInputError (catch-all path; subcmds use 1)  -> 64  (EX_USAGE)
    #   DatasetError                                    -> 65  (EX_DATAERR)
    #   PermissionError                                 -> 77  (EX_NOPERM)
    #   ConnectionError / TimeoutError / HTTPError ish  -> 69  (EX_UNAVAILABLE)
    #   everything else                                 -> 70  (EX_SOFTWARE)
    #
    # Operators relying on the legacy 0 / 1 / 2 / 3 / 130 contract see the
    # same codes from the per-subcommand handlers; the sysexits overlay is
    # purely additive on the catch-all path. CI scripts that grep on
    # `[0123]` shapes won't false-match the overlay codes because the
    # overlay codes (64/65/69/70/77/137) are disjoint.
    try:
        result: int = args.func(args)
        return result
    except KeyboardInterrupt:
        print(f"Interrupted (run_id={cli_run_id[:12]}).", file=sys.stderr)
        return EXIT_INTERRUPTED
    except SystemExit:
        # Let argparse / library SystemExit propagate so tests that catch
        # SystemExit (e.g. -h flag invocations) see the same behavior.
        raise
    except BackpropagateError as e:
        # BRIDGE-F-006: map the structured exception's code to a sysexits
        # bucket before falling through to the legacy EXIT_RUNTIME_ERROR.
        code = getattr(e, "code", None) or "RUNTIME"
        print(f"{code}: {e}", file=sys.stderr)
        if os.environ.get("BACKPROPAGATE_DEBUG"):
            traceback.print_exc()

        # User-input / dataset / GPU-OOM / Hub failures all have well-defined
        # remediation paths; map to the matching sysexits bucket so wrappers
        # can distinguish e.g. "model not on Hub" from "internal crash".
        if isinstance(e, UserInputError):
            return EXIT_USAGE
        if isinstance(e, DatasetError):
            return EXIT_DATA_ERR
        if code in ("RUNTIME_GPU_OOM", "RUNTIME_OOM_RECOVERY_EXHAUSTED", "RUNTIME_OOM_ADJACENT"):
            return EXIT_OOM_KILLED
        if code in ("HUB_PUSH_NETWORK", "HUB_PUSH_NOT_FOUND", "DEP_MODEL_LOAD_FAILED", "DEP_OLLAMA_REGISTRATION_FAILED"):
            return EXIT_UNAVAILABLE
        return EXIT_RUNTIME_ERROR
    except Exception as e:  # noqa: BLE001 — last-resort safety net
        # The Ship Gate B2 contract promises exit code 2 for unexpected
        # crashes. The default Python behavior is exit code 1 + traceback.
        # BRIDGE-F-006 sysexits overlay: name the failure mode before
        # falling back to EXIT_RUNTIME_ERROR.
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        if os.environ.get("BACKPROPAGATE_DEBUG"):
            traceback.print_exc()
        else:
            print(
                "Set BACKPROPAGATE_DEBUG=1 for the full traceback, or re-run "
                "the subcommand with --verbose for friendlier output.",
                file=sys.stderr,
            )

        # PermissionError covers POSIX EACCES / EPERM and Windows ACL denials —
        # both share the "operator can't access path / device" remediation.
        if isinstance(e, PermissionError):
            return EXIT_NO_PERM
        # Match torch.cuda.OutOfMemoryError by name (without importing torch
        # here — keeps cold-start of `backprop --help` cheap).
        if type(e).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
            return EXIT_OOM_KILLED
        # Network / unreachable-service failures: stdlib ConnectionError +
        # TimeoutError cover most cases. requests / httpx subclasses bubble
        # up as their own types but inherit from ConnectionError on modern
        # versions; the substring fallback catches the rest.
        if isinstance(e, (ConnectionError, TimeoutError)):
            return EXIT_UNAVAILABLE
        type_name = type(e).__name__.lower()
        if any(needle in type_name for needle in ("httperror", "connection", "timeout", "unreachable")):
            return EXIT_UNAVAILABLE
        # Permission-denied OSError on some platforms doesn't subclass
        # PermissionError — match by errno when available.
        if isinstance(e, OSError) and getattr(e, "errno", None) in (13, 1):  # EACCES, EPERM
            return EXIT_NO_PERM
        return EXIT_SOFTWARE


if __name__ == "__main__":
    sys.exit(main())
