"""
Backpropagate - Reflex state classes
=====================================

The ``rx.State`` subclasses that drive the five UI surfaces (Train, Multi-Run,
Export, Dataset, Runs). ``TrainState`` / ``MultiRunState`` / ``ExportState``
remain pre-Phase-3 stubs for the backend-write side (Start training is still a
placeholder until ``Trainer`` integration lands); ``DatasetState`` is real
(file upload + size-cap streaming + validator) as of v1.2.0; ``RunsState`` is
real (RunHistoryManager-backed read of on-disk run JSON) as of v1.2.0
(FRONTEND-F-RUN-HISTORY-PAGE).

The state classes are intentionally split (not one mega-class) so each
surface's WebSocket bundle stays small. Reflex coalesces the per-class state
into a single client connection automatically.

Config fields shared by ``TrainState`` and ``MultiRunState`` (Model, Training
shape, LoRA, Dataset) are duplicated by design: Reflex's State metaclass only
auto-registers event handlers DIRECTLY declared on the ``rx.State`` subclass
(plain Python mixin inheritance is invisible to the framework's event-trigger
scan, per 2026-05-22 verification). The setters route through module-level
``_apply_*`` helpers so the duplicated fields share one validation contract.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Literal

import reflex as rx

# Shared literal types — referenced across multiple State classes.
RunState = Literal["idle", "loading", "active", "paused", "done", "stopped", "error"]
Theme = Literal["dark", "light"]
ActiveSurface = Literal["train", "multi-run", "export", "dataset"]
ExportFormat = Literal["lora", "merged", "gguf"]
Quantization = Literal["4-bit", "8-bit", "16-bit"]
# ui-v2 P3: the training mode picker. qlora = LoRA on a 4-bit base (the CLI
# default), lora = LoRA on a 16-bit base (--no-4bit), full = --mode full.
TrainMode = Literal["qlora", "lora", "full"]
Method = Literal["sft", "orpo", "simpo", "kto"]
# ui-v2 P2: exactly what the multi-run job maps to CLI flags
# (slao | simple -> --merge-mode; ties -> --merge-mode slao --merge-strategy ties).
MergeMode = Literal["slao", "simple", "ties"]
# ui-v2 P2: exactly `backprop export --quantization` choices.
GgufQuant = Literal["f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k"]
DatasetFormatHint = Literal["auto", "sharegpt", "alpaca", "openai", "jsonl"]

# Constants for the setters' clamps. Centralised so an operator can read the
# bounds in one place — they also appear in the operator-facing error strings.
_STEPS_MIN, _STEPS_MAX = 1, 100_000
_LR_MIN, _LR_MAX = 1e-7, 1.0
_LORA_R_MIN, _LORA_R_MAX = 1, 256
_LORA_ALPHA_MIN, _LORA_ALPHA_MAX = 1, 512
_LORA_DROPOUT_MIN, _LORA_DROPOUT_MAX = 0.0, 1.0
_GPU_TEMP_MIN, _GPU_TEMP_MAX = 50, 105  # the CLI's --gpu-max-temp range
_METHOD_PARAM_MAX = 100.0
_NUM_RUNS_MIN, _NUM_RUNS_MAX = 1, 100
_SAMPLES_PER_RUN_MIN, _SAMPLES_PER_RUN_MAX = 1, 1_000_000
_TOKENS_MIN, _TOKENS_MAX = 0, 1_000_000

# Comma-separated identifier list (LoRA target modules). The character set is
# strict on purpose — anything outside it cannot resolve to a real attention
# module name and the only reason to type it is mistake or injection probe.
_TARGET_MODULES_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.,\s]*$")

# W&B run name: same shape that wandb itself accepts.
_WANDB_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._\-]+$")



def _ts_now() -> str:
    """HH:MM:SS timestamp for event-log rows (ui-v2 event feed)."""
    import datetime as _dt

    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%H:%M:%S")


def _merge_job_history_rows(
    rows: list[dict], history_dir: object, *, status: str | None, limit: int
) -> list[dict]:
    """Merge per-job histories into the root run-history rows.

    UI-spawned training (JobManager) writes its run history to
    ``<ui-output>/jobs/<run_id>/output/run_history.json`` — a tree the root
    RunHistoryManager never reads, so UI-driven runs were invisible on the
    Runs page even after two successful trainings (ui-v2 P1 fix round).
    De-dupes by run_id, re-applies the status filter to the job entries,
    sorts by ``started_at`` descending, and re-trims to ``limit``.
    """
    import json as _json
    from pathlib import Path as _Path

    def _sort_trim(all_rows: list[dict]) -> list[dict]:
        all_rows.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
        return all_rows[:limit]

    seen = {str(r.get("run_id") or "") for r in rows}
    merged = list(rows)
    jobs_root = _Path(str(history_dir)) / "jobs"
    if not jobs_root.is_dir():
        return _sort_trim(merged)
    for hist in sorted(jobs_root.glob("*/output/run_history.json")):
        try:
            data = _json.loads(hist.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            continue
        entries = data.get("runs") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rid = str(entry.get("run_id") or "")
            if rid and rid in seen:
                continue
            override = _job_status_override(hist.parent.parent)
            if override:
                entry = {**entry, "status": override}
            if status and str(entry.get("status") or "") != status:
                continue
            if rid:
                seen.add(rid)
            merged.append(entry)
    return _sort_trim(merged)


def _job_status_override(job_dir: object) -> str | None:
    """A UI job's own outcome from ``job.json`` (ui-v2 P2).

    RunHistoryManager records a cooperative stop as "completed" (training
    returned normally); the job record knows it was stopped. Returns
    "stopped" / "failed" when the job record says so, else None.
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        job = _json.loads((_Path(str(job_dir)) / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    status = str(job.get("status") or "") if isinstance(job, dict) else ""
    return status if status in ("stopped", "failed") else None


def _locate_run_history_dir(history_dir: object, run_id: str):
    """The directory whose run_history.json holds ``run_id``.

    CLI runs live in the UI output dir's own history; UI-started runs live
    in ``jobs/<job>/output/run_history.json`` (ui-v2 P1/P2). Returns
    ``(dir, job_dir_or_None)``; falls back to ``(history_dir, None)``.
    """
    import json as _json
    from pathlib import Path as _Path

    root = _Path(str(history_dir))
    if not run_id:
        return root, None
    for hist in sorted((root / "jobs").glob("*/output/run_history.json")):
        try:
            data = _json.loads(hist.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = data.get("runs") if isinstance(data, dict) else data
        if isinstance(entries, list) and any(
            isinstance(e, dict) and str(e.get("run_id") or "") == run_id for e in entries
        ):
            return hist.parent, hist.parent.parent
    return root, None


def _job_loss_curve(job_dir: object, limit: int = 400) -> list[float]:
    """Loss values from a UI job's events.jsonl (step rows), oldest first."""
    import json as _json
    from pathlib import Path as _Path

    out: list[float] = []
    try:
        lines = (_Path(str(job_dir)) / "events.jsonl").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            row = _json.loads(line)
        except ValueError:
            continue
        loss = row.get("loss") if isinstance(row, dict) and row.get("kind") == "step" else None
        if isinstance(loss, (int, float)):
            out.append(float(loss))
    return out[-limit:]


def _fmt_started(value: object) -> str:
    """'2026-10-02T04:45:42.269557' -> '2026-10-02 04:45' (the Runs table)."""
    from datetime import datetime

    text = str(value or "").strip()
    if not text or text == "-":
        return "-"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:16].replace("T", " ")


def _dataset_label(value: object) -> str:
    """A dataset's file name (or the HF dataset id). Full local paths stay
    off the client: the name is what identifies it, and it carries no
    home directory."""
    import re as _re

    text = str(value or "").strip()
    if not text or text == "-":
        return "-"
    if "/" in text or "\\" in text:
        parts = [part for part in _re.split(r"[\\/]", text) if part]
        # "org/name" (HF dataset id) stays as is; a filesystem path -> its name.
        if len(parts) == 2 and not _re.match(r"^[A-Za-z]:$", parts[0]) and not text.startswith(("/", "~", ".")):
            return text
        return parts[-1] if parts else text
    return text


def _fmt_eta_range(eta_s: float) -> tuple[str, str]:
    """Format an ETA as a low/high pair (ui-v2 requirement 10).

    ETAs on live training are approximate — step times vary with compile,
    eval, and checkpoint saves — so the UI quotes a range (EMA -20% …
    EMA +30%) instead of a fake-exact countdown.
    """

    def fmt(seconds: float) -> str:
        if seconds < 90:
            return f"{int(max(5, round(seconds / 5.0) * 5))} s"
        if seconds < 5400:
            return f"{int(round(seconds / 60.0))} min"
        return f"{seconds / 3600.0:.1f} h"

    return fmt(max(1.0, eta_s * 0.8)), fmt(max(2.0, eta_s * 1.3))




# ---------------------------------------------------------------------------
# Path validation helper — FRONTEND-A-002 fix
# ---------------------------------------------------------------------------
#
# All user-supplied path fields on the four state classes flow through this
# helper so the FB-003 + F-002 hardening in ``ui_security`` is exercised by
# the Reflex surface (it was previously dead code there). The helper resolves
# the value against ``get_ui_output_dir()`` as the allowed base; absolute
# escapes and ``..`` traversal raise ``PathTraversalError`` which the caller
# surfaces as the ``*_error`` companion field.
#
# Empty strings are allowed (they represent "not yet set"). The validator is
# deliberately lenient on relative paths that resolve INSIDE the allowed
# base — operators routinely drop in ``runs/run-x/adapter`` style paths.


def _validate_ui_path(value: str) -> tuple[str, str]:
    """Validate a user-supplied path against the UI output dir.

    Returns a ``(cleaned_value, error_message)`` tuple. On success the error
    is the empty string; on failure ``cleaned_value`` is the empty string and
    the error carries a short operator-facing message. Empty input is a
    pass-through (no error, no value) so the input field can be cleared.
    """
    if not value or not value.strip():
        return "", ""
    candidate = value.strip()
    # A NUL byte can never be part of a legitimate path. Some platform / Python
    # combinations (e.g. Windows, 3.13) resolve it without raising, so refuse it
    # explicitly instead of relying on ``Path.resolve`` to.
    if "\x00" in candidate:
        return "", "Invalid path: contains a NUL byte."
    try:
        from .security import safe_path
        from .ui_security import get_ui_output_dir

        base = get_ui_output_dir()
        resolved = safe_path(candidate, allowed_base=base, allow_relative=True)
        return str(resolved), ""
    except Exception as exc:  # noqa: BLE001 — surface as operator-facing string
        return "", f"Invalid path: {exc}"


# ---------------------------------------------------------------------------
# Run-id validation helper — UI-A-003 fix (Wave A1 HIGH)
# ---------------------------------------------------------------------------
#
# Run IDs originate from two user-controlled surfaces: the dynamic route
# param ``rid`` (``/runs/[rid]``) and the ``diff_other_run_id`` text input.
# Both flow into ``subprocess`` argv for the diff-runs shell-out (the only
# action that still shells out — Replay / Delete / Export run in-process via
# RunHistoryManager). An option-shaped value (e.g. ``--to=/etc/passwd`` or
# ``-o``) is parsed by the downstream argparse-based CLI as a FLAG, not a
# positional run id — letting a remote operator (under the documented
# ``--share + --auth`` flow) smuggle arbitrary flags into the spawned process.
#
# Real run IDs are UUID-hex / wandb-style slugs: ``[A-Za-z0-9_-]``. We pin a
# strict allowlist (1-64 chars, no leading ``-``) at the trust boundary so a
# malformed value is rejected with a clean error before it can reach argv.
# The leading char is constrained to ``[A-Za-z0-9_]`` (NOT ``-``) so the
# value can never be parsed as a CLI flag even before the ``--`` separator.
# (The ``--`` end-of-options separator inserted in the diff-runs shell-out is
# the defense-in-depth second layer; this regex is the first.)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")


def _validate_run_id(value: str) -> tuple[str, str]:
    """Validate a user-supplied run id against the strict allowlist.

    Returns a ``(cleaned_value, error_message)`` tuple mirroring
    ``_validate_ui_path``: on success the error is empty; on failure the
    value is empty and the error carries a short operator-facing message.
    Empty input is a pass-through (no error, no value).

    The allowlist (``^[A-Za-z0-9_-]{1,64}$``) rejects any leading ``-`` (so
    the value can never be parsed as a CLI flag), path separators, dots
    (no ``..`` traversal), whitespace, and shell metacharacters.
    """
    if not value or not value.strip():
        return "", ""
    candidate = value.strip()
    if not _RUN_ID_RE.match(candidate):
        return "", (
            "Invalid run id — expected 1-64 chars of letters, digits, "
            "'_' or '-' (no leading '-', no path separators)."
        )
    return candidate, ""


# ---------------------------------------------------------------------------
# HF token-file path validation helper — UI-A-001 fix
# ---------------------------------------------------------------------------
#
# The HF token-file path field (``set_hub_token_file_path``, mirroring the
# ``--token-file`` CLI flag) is a path to READ an EXISTING user-owned secret
# file — NOT a path the UI writes into. It must NOT be validated against the
# UI OUTPUT sandbox (``get_ui_output_dir()`` → ``~/.backpropagate/ui-outputs``):
# that sandbox is mutually exclusive with the documented / placeholder token
# location ``~/.config/backpropagate/hf-token`` (``~/.config`` sits on the
# UI-output forbidden-base denylist). Routing this READ path through the WRITE
# sandbox made the documented path impossible to enter and the shipped
# ``--token-file`` UI field unusable.
#
# The correct validation mirrors the CLI's ``_read_hub_token_file``
# (cli.py ~1115): expanduser + resolve, then require the target to be an
# EXISTING regular file (a credential the operator already created). We add a
# basic hostile-input guard (reject NUL bytes) and reject directories /
# missing files with clear, operator-facing messages. The file is NEVER read
# or written here — only its shape is validated; ``push_to_hub`` reads it at
# push time via the CLI helper so the mode-0600 warning + empty-file error
# stay in one place. Standard user locations (``~/.config/...``,
# ``~/.hf-token``, an absolute path under the operator's home, etc.) are
# allowed by design — this is the operator's own credential, not a UI write.
def _validate_token_file_path(value: str) -> tuple[str, str]:
    """Validate a user-supplied HF token-FILE path for READING a credential.

    Returns a ``(cleaned_value, error_message)`` tuple mirroring
    ``_validate_ui_path`` / ``_validate_run_id``: on success the error is the
    empty string and ``cleaned_value`` is the resolved path; on failure
    ``cleaned_value`` is the empty string and the error carries a short
    operator-facing message. Empty input is a pass-through (no error, no
    value) so the field can be cleared.

    Unlike ``_validate_ui_path`` this does NOT constrain the path to the UI
    output sandbox — the token file is a pre-existing user-owned secret the
    operator points us at (e.g. ``~/.config/backpropagate/hf-token``), not a
    UI write target. We require an existing regular file and reject NUL bytes,
    directories, and missing paths.
    """
    if not value or not value.strip():
        return "", ""
    candidate = value.strip()
    # Reject obviously hostile input before touching the filesystem.
    if "\x00" in candidate:
        return "", "Invalid token-file path: contains a NUL byte."
    try:
        path = Path(candidate).expanduser()
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            # Fall back to the un-resolved expansion if resolve() blows up
            # (e.g. a symlink loop); existence checks below still apply.
            resolved = path
        if not resolved.exists():
            return "", (
                f"Token-file path does not exist: {resolved}. Create the file "
                "(e.g. `printf 'hf_xxx' > ~/.config/backpropagate/hf-token`) "
                "and point this field at it."
            )
        if not resolved.is_file():
            return "", (
                f"Token-file path is not a regular file: {resolved}. "
                "Point this field at the token FILE, not a directory."
            )
        return str(resolved), ""
    except (OSError, ValueError) as exc:
        return "", f"Invalid token-file path: {exc}"


# ---------------------------------------------------------------------------
# Action-string redaction — V2-a fix (Wave A2 verifier gap)
# ---------------------------------------------------------------------------
#
# RunDetailState's in-process action handlers (delete_run / export_run /
# replay, rewired in Wave A1 from broken subprocess shell-outs) build
# operator-facing strings that embed RAW absolute paths — the sandbox
# ``history_dir`` / ``out_path`` (both under ``~/.backpropagate/ui-outputs``,
# which contains the OS home dir + username) and the bare ``{exc}`` repr of an
# OSError (``[Errno 2] ... '/home/<user>/...'``). Those strings are assigned to
# ``action_error`` / ``action_result``, which are PUBLIC (client-serialized)
# Reflex vars — under the documented ``--share + --auth`` flow they ship to the
# remote browser, leaking the operator's home dir + username (the exact FB-011
# class ``sanitize_error_for_user`` exists to prevent). Route every
# path-bearing action string through this helper before assignment so the home
# prefix is replaced with ``<redacted-path>``.
def _redact_action(text: str) -> str:
    """Redact absolute filesystem paths from an operator-facing action string.

    Thin wrapper over ``ui_security._redact_paths`` (the same redactor
    ``sanitize_error_for_user`` uses) so the in-process action handlers can
    scrub ``history_dir`` / ``out_path`` / raw ``OSError`` reprs out of the
    client-serialized ``action_error`` / ``action_result`` vars. Falls back to
    returning the text unchanged if the import fails (the redactor module is
    framework-agnostic and always importable, but be defensive — a redaction
    failure must never crash the handler).
    """
    try:
        from .ui_security import _redact_paths

        return _redact_paths(text)
    except Exception:  # noqa: BLE001 — redaction is best-effort, never fatal
        return text


def _coerce_int(value: object) -> int | None:
    """Best-effort ``int`` cast; returns ``None`` if value can't be parsed.

    The Reflex setter signature is ``str | int | float`` depending on the
    input widget, so a single helper keeps the per-field setter terse.
    """
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # nan / +-inf cannot become an int (int() raises ValueError /
        # OverflowError inside the event handler) - treat as a parse failure.
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
            except ValueError:
                return None
            # "inf" / "1e999" parse as floats but are not usable integers.
            return int(f) if math.isfinite(f) else None
    return None


def _coerce_float(value: object) -> float | None:
    """Best-effort ``float`` cast; ``None`` if unparseable or NaN.

    NaN is rejected because it compares False against every bound, so it would
    sail through ``_clamp_float`` and be stored. ``+-inf`` is kept: the clamp
    turns it into the nearest bound.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return None if math.isnan(f) else f
    return None


def _clamp_int(name: str, raw: object, lo: int, hi: int) -> tuple[int | None, str]:
    """Parse + clamp an integer. Returns ``(value, error)``.

    On parse failure ``value`` is ``None`` and the caller should keep the
    previous state. Out-of-range values are clamped silently INSIDE the
    range — the error string carries the operator-facing nudge so the UI
    can surface it via the ``*_error`` companion.
    """
    n = _coerce_int(raw)
    if n is None:
        return None, f"{name} must be an integer (got {raw!r})"
    if n < lo:
        return lo, f"{name} clamped to minimum {lo} (was {n})"
    if n > hi:
        return hi, f"{name} clamped to maximum {hi} (was {n})"
    return n, ""


def _clamp_float(
    name: str, raw: object, lo: float, hi: float
) -> tuple[float | None, str]:
    f = _coerce_float(raw)
    if f is None:
        return None, f"{name} must be a number (got {raw!r})"
    if f < lo:
        return lo, f"{name} clamped to minimum {lo:g} (was {f:g})"
    if f > hi:
        return hi, f"{name} clamped to maximum {hi:g} (was {f:g})"
    return f, ""


class AppState(rx.State):
    """Top-level state: theme toggle, active surface, current run_id.

    Cross-surface state lives here so the header / left nav / footer can read
    it without coupling to a specific page's state class.
    """

    theme: Theme = "dark"
    active_surface: ActiveSurface = "train"
    run_id: str = ""

    @rx.event
    def toggle_theme(self) -> None:
        """Flip dark/light (deprecated path — kept for back-compat).

        FRONTEND-F-001 (Wave 5.5): the load-bearing theme toggle now lives
        on Reflex's built-in ``rx.color_mode`` + ``rx.toggle_color_mode``
        plumbing — the header button binds to those directly so the DOM
        actually mutates (Radix theme provider writes ``class="light"`` /
        ``class="dark"`` on the html root, which fires the ``.light`` /
        ``.light-theme`` selector in ``ui_theme.TOKENS_CSS``).

        This server-side ``AppState.theme`` field + handler stay for
        backward compatibility — external automation or future surfaces
        may still introspect/mutate it — but they no longer drive the
        visible theme. The v1.2 bug was that ONLY this server-side
        toggle existed, so the icon swapped but the page stayed dark.
        """
        self.theme = "light" if self.theme == "dark" else "dark"

    @rx.event
    def set_active_surface(self, surface: str) -> None:
        """Left-nav click handler. The ``surface`` arg is a free string from
        the click event; clamp to the known set."""
        if surface in ("train", "multi-run", "export", "dataset"):
            self.active_surface = surface  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared training-config setter helpers (FRONTEND-A-005)
# ---------------------------------------------------------------------------
#
# Both TrainState and MultiRunState carry the same Model / Training-shape /
# LoRA config fields. Reflex's State metaclass only auto-registers event
# handlers that are DIRECTLY declared on the rx.State subclass (mixin
# inheritance is invisible to the framework's event-trigger scan, per
# 2026-05-22 verification), so we cannot share via a plain Python mixin.
# The fields and setters are therefore duplicated by design — but the
# logic each setter calls into is centralised in the helpers below so the
# behaviour stays in lockstep.
#
# When the Phase 3 Trainer hookup lands, both TrainState and MultiRunState
# will read from the same config snapshot helper, so an operator typing on
# one surface still sees the value carried to the other (the Trainer
# accepts a single config struct, regardless of which surface fired it).


def _apply_model(value: str) -> tuple[str, str]:
    """Validation logic for the HF model id / local model path setter."""
    if value and ("/" in value or "\\" in value) and Path(value).is_absolute():
        cleaned, err = _validate_ui_path(value)
        return cleaned, err
    return value, ""


def _apply_batch_size(value: object) -> tuple[str | None, str]:
    """``'auto'`` OR a positive int.

    Returns ``(stored_str_or_none, error)``. On parse failure the caller
    keeps the previous value and surfaces the error string.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto", ""
    n = _coerce_int(value)
    if n is None:
        return None, (
            f"Batch size must be 'auto' or a positive integer (got {value!r})"
        )
    if n < 1:
        return "1", "Batch size clamped to minimum 1"
    if n > 4096:
        return "4096", "Batch size clamped to maximum 4096"
    return str(n), ""


def _apply_target_modules(value: str) -> tuple[str | None, str]:
    """Comma-separated identifier list with strict character set + 32-item cap.

    Returns ``(canonical_string_or_none, error)``. ``None`` means parse
    failure — caller keeps the previous value.
    """
    if not value or not value.strip():
        return "", ""
    cleaned = value.strip()
    if cleaned.lower() == "all-linear":
        return "all-linear", ""
    if not _TARGET_MODULES_RE.match(cleaned):
        return None, (
            "Target modules: only letters, digits, underscore, comma, "
            "whitespace allowed"
        )
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return "", "Target modules cannot be empty"
    if len(parts) > 32:
        return None, "At most 32 target modules per LoRA"
    return ", ".join(parts), ""


def _apply_wandb_run_name(value: str) -> tuple[str | None, str]:
    if not value or not value.strip():
        return "", ""
    cleaned = value.strip()
    if len(cleaned) > 128:
        return None, "Run name too long (max 128 chars)"
    if not _WANDB_RUN_NAME_RE.match(cleaned):
        return None, "Run name: alphanumerics, dot, underscore, dash only"
    return cleaned, ""


def _read_log_tail(log_path: str, n: int = 20) -> list[str]:
    """Last ``n`` non-empty lines of a job's output.log (requirement 10:
    failures show the log next to the error code). Never raises."""
    if not log_path:
        return []
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 32768))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    # tqdm redraws with carriage returns; keep only the last frame per line.
    lines = [ln.split("\r")[-1].rstrip() for ln in text.splitlines()]
    return [ln[:300] for ln in lines if ln.strip()][-n:]


def _fmt_bytes(n: float) -> str:
    """Human size: "27.6 GB", "519.6 MB", "12 KB" (binary units, like the OS)."""
    n = float(n or 0)
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}" if unit != "KB" else f"{n / scale:.0f} KB"
    return f"{n:.0f} B"


def _fmt_local_time(ts: float | None = None) -> str:
    """"14:05" local time for "updated at" captions."""
    import datetime as _dt

    when = _dt.datetime.fromtimestamp(ts) if ts else _dt.datetime.now()
    return when.strftime("%H:%M")


# ---- ui-v2 P3: presets, LoRA shapes, methods, the VRAM estimate ----------------

#: LoRA shape quick picks: (rank, alpha, target modules). Exactly
#: ``config.LORA_PRESETS`` -- "quality" is the CLI default.
LORA_SHAPES: dict[str, tuple[int, int, str]] = {
    "quality": (256, 512, "all-linear"),
    "fast": (16, 32, "q_proj, v_proj"),
}

#: Method knob defaults (the config defaults the CLI falls back to).
METHOD_DEFAULTS: dict[str, float] = {
    "orpo_beta": 0.1,
    "simpo_beta": 2.0,
    "simpo_gamma": 1.0,
    "kto_beta": 0.1,
    "kto_desirable_weight": 1.0,
    "kto_undesirable_weight": 1.0,
}
_METHOD_KEYS: dict[str, tuple[str, ...]] = {
    "sft": (),
    "orpo": ("orpo_beta",),
    "simpo": ("simpo_beta", "simpo_gamma"),
    "kto": ("kto_beta", "kto_desirable_weight", "kto_undesirable_weight"),
}

#: What each method needs in the dataset (shown under the Dataset path).
METHOD_DATA_HINTS: dict[str, str] = {
    "sft": "Conversations or instructions. Format detected from contents: "
    "Alpaca, ShareGPT, OpenAI or raw JSONL.",
    "orpo": "Preference pairs: each row has prompt, chosen and rejected.",
    "simpo": "Preference pairs: each row has prompt, chosen and rejected.",
    "kto": "Unpaired feedback: each row has prompt, completion and a true/false label.",
}


def model_preset_options() -> list[dict[str, str]]:
    """The model presets for the "Start from" picker (``config.MODEL_PRESETS``)."""
    try:
        from .config import MODEL_PRESETS
    except Exception:  # noqa: BLE001 -- config import failure: custom only
        return []
    out: list[dict[str, str]] = []
    for key, p in MODEL_PRESETS.items():
        label = str(p.description).split(" — ")[0].strip() or p.model_id
        note = f"{p.best_for} License: {p.license}."
        restriction = getattr(p, "license_restriction", None)
        if restriction:
            note += " " + str(restriction).strip()
        out.append(
            {
                "key": key,
                "label": label,
                "model_id": p.model_id,
                "lora_r": str(p.recommended_lora_r),
                "note": note,
            }
        )
    return out


def _preset_for_model(model: str) -> str:
    for opt in model_preset_options():
        if opt["model_id"].lower() == (model or "").strip().lower():
            return opt["key"]
    return "custom"


def _lora_shape_of(r: int, alpha: int, targets: str) -> str:
    for name, (sr, sa, st) in LORA_SHAPES.items():
        if (r, alpha, targets) == (sr, sa, st):
            return name
    return "custom"


def _training_spec_fields(form) -> dict:
    """JobSpec fields shared by the Single run and Multi-run forms.

    ``form`` is a TrainState / MultiRunState (same field names). Every value
    maps to a real CLI flag (ui_jobs._training_flags).
    """
    mode = str(form.train_mode)
    method = str(form.method)
    fields: dict = {
        "model": form.model,
        "dataset_path": form.dataset_path,
        "steps": int(form.steps),
        "batch": str(form.batch_size),
        "lr": float(form.learning_rate),
        "lora_r": int(form.lora_r),
        "mode": "full" if mode == "full" else "lora",
        "base_4bit": mode != "lora",
        "method": method,
        "method_params": {k: float(getattr(form, k)) for k in _METHOD_KEYS.get(method, ())},
        "run_name": str(form.wandb_run_name or ""),
        "gpu_max_temp": float(form.gpu_temp_threshold),
        "gradient_checkpointing": bool(form.gradient_checkpointing),
    }
    if mode != "full":
        fields["lora_alpha"] = int(form.lora_alpha)
        fields["lora_dropout"] = float(form.lora_dropout)
        fields["target_modules"] = str(form.target_modules).replace(" ", "")
    return fields


def _form_errors(form) -> list[str]:
    return [
        err
        for err in (
            form.model_error,
            form.dataset_path_error,
            form.steps_error,
            form.batch_size_error,
            form.learning_rate_error,
            form.lora_r_error,
            form.lora_alpha_error,
            form.lora_dropout_error,
            form.target_modules_error,
            form.method_param_error,
            form.gpu_temp_threshold_error,
            form.wandb_run_name_error,
        )
        if err
    ]


VRAM_VERDICT_TITLES = {
    "fits": "Fits",
    "tight": "Tight",
    "wont_fit": "Won't fit",
    "unknown": "No estimate",
}


class TrainState(rx.State):
    """Train surface state: config + live run progress.

    Config fields are duplicated with MultiRunState by design (Reflex's
    metaclass requires direct declaration); validation routes through the
    module-level ``_apply_*`` helpers so the two classes share one
    behaviour spec.
    """

    # ---- Configuration form ------------------------------------------------
    # ui-v2 P3: defaults are the CLI's (`backprop train` with no flags):
    # Qwen 2.5 7B, QLoRA, SFT, the "quality" LoRA shape (r 256, alpha 512,
    # all linear layers).
    preset: str = "qwen2.5-7b"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_error: str = ""
    dataset_path: str = ""
    dataset_path_error: str = ""
    steps: int = 100
    steps_error: str = ""
    batch_size: str = "auto"
    batch_size_error: str = ""
    learning_rate: float = 2e-4
    learning_rate_error: str = ""
    lora_r: int = 256
    lora_r_error: str = ""
    lora_alpha: int = 512
    lora_alpha_error: str = ""
    lora_dropout: float = 0.05
    lora_dropout_error: str = ""
    target_modules: str = "all-linear"
    target_modules_error: str = ""
    train_mode: TrainMode = "qlora"
    method: Method = "sft"
    orpo_beta: float = 0.1
    simpo_beta: float = 2.0
    simpo_gamma: float = 1.0
    kto_beta: float = 0.1
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
    method_param_error: str = ""

    # ---- Advanced flags ----------------------------------------------------
    # --gpu-max-temp: stop and save when the GPU stays above this.
    gpu_temp_threshold: int = 90
    gpu_temp_threshold_error: str = ""
    wandb_run_name: str = ""
    wandb_run_name_error: str = ""
    gradient_checkpointing: bool = True

    # ---- Inline VRAM estimate (P3; `backprop estimate-vram` numbers) --------
    vram_est_verdict: str = "unknown"
    vram_est_total: float = 0.0
    vram_est_batch: int = 0
    vram_est_note: str = ""
    vram_est_source: str = "estimate"  # "measured" once calibrated on this GPU
    # Latest refresh wins: refreshes run concurrently off the event loop and
    # the first one (which imports the trainer module) can finish last.
    _vram_est_seq: int = 0
    # Why the last run stopped when the GPU temperature limit tripped.
    job_safety_reason: str = ""

    # ---- Live run progress -------------------------------------------------
    run_state: RunState = "idle"
    current_step: int = 0
    current_loss: float = 0.0
    loss_history: list[float] = []
    # ui-v2 P1 fix round: parallel series for the chart + smoothed readout.
    # ``step_history[i]`` is the trainer's global_step for loss_history[i]
    # (real numbers, not enumerate index, so the x-axis tracks training even
    # when logging_steps > 1); ``ema_history`` holds the debiased EMA value
    # (None-safe: recharts skips null points with connectNulls off).
    ema_history: list[float] = []
    step_history: list[int] = []
    # True while a stop request is in flight (control.json written, child
    # still saving/exiting). Disables the Stop button against double-clicks.
    stop_requested: bool = False
    # Requirement 10: "last step N s ago" heartbeat + ETA range label.
    # Poller-owned (1 Hz during active runs); frozen when the run ends.
    heartbeat_label: str = ""
    eta_label: str = ""
    gpu_temp: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0

    # CLIUI-B-001's ``cli_notice`` pointer field was removed in the ui-v2 P1
    # fix round: UI-driven training is real now, so the "use the shell"
    # notice can never fire on this surface. MultiRunState/ExportState keep
    # theirs until P2 wires those pages into the JobManager.

    # Event log — each entry is a dict with keys: t (timestamp str),
    # level (one of info/ok/warn/err/tx/hf), msg (str).
    events: list[dict] = []

    # ---- Computed Vars (FRONTEND-6 Wave 6b) --------------------------------

    @rx.var
    def loss_chart_data(self) -> list[dict]:
        """Shape the step/loss/EMA series for ``rx.recharts.line_chart``.

        Returns ``[{"step": s, "loss": v, "ema": e}, ...]`` — real trainer
        step numbers on x (not index), raw loss + debiased EMA on y so the
        chart draws the design-digest "raw faint + smoothed bold" pairing.
        """
        out: list[dict] = []
        for i, loss in enumerate(self.loss_history):
            step = self.step_history[i] if i < len(self.step_history) else i
            row: dict = {"step": step, "loss": round(float(loss), 5)}
            if i < len(self.ema_history):
                row["ema"] = round(float(self.ema_history[i]), 5)
            out.append(row)
        return out

    @rx.var
    def loss_label(self) -> str:
        """Current loss formatted for display (3-4 decimals, not 17)."""
        if not self.loss_history:
            return ""
        return f"{self.current_loss:.4f}"

    @rx.var
    def ema_loss_label(self) -> str:
        """Smoothed loss (debiased EMA), the headline number on the chart card."""
        if not self.ema_history:
            return ""
        return f"{self.ema_history[-1]:.4f}"

    @rx.var
    def job_chip_label(self) -> str:
        """The progress card's state chip."""
        return {
            "multi_run": "MULTI-RUN",
            "export": "EXPORTING",
            "calibrate": "MEASURING",
        }.get(self.job_kind, "TRAINING")

    @rx.var
    def job_run_label(self) -> str:
        """'run 2 of 3' while a multi-run is going; '' otherwise."""
        if self.job_kind == "multi_run" and self.job_runs:
            return f"run {max(self.job_run, 1)} of {self.job_runs}"
        return ""

    @rx.var
    def job_has_steps(self) -> bool:
        """Export and calibration have phases, not steps: the card shows the
        phase instead."""
        return self.job_kind not in ("export", "calibrate")

    @rx.var
    def job_is_training(self) -> bool:
        """True for jobs that leave a trained model (single run, multi-run)."""
        return self.job_kind not in ("export", "calibrate")

    @rx.var
    def done_title(self) -> str:
        """Heading of the post-job panel."""
        if self.job_kind == "export":
            return "Export complete"
        if self.job_kind == "calibrate":
            return (
                "Measured on this GPU" if self.run_state == "done" else "Measurement stopped"
            )
        if self.run_state == "stopped":
            return "Stopped · what next?"
        return "Run complete · what next?"

    @rx.var
    def form_disabled(self) -> bool:
        """True while a run is active — config fields lock during training."""
        return self.run_state == "active"

    @rx.var
    def lora_shape(self) -> str:
        """quality | fast | custom, from the current rank / alpha / targets."""
        return _lora_shape_of(self.lora_r, self.lora_alpha, self.target_modules)

    @rx.var
    def preset_note(self) -> str:
        for opt in model_preset_options():
            if opt["key"] == self.preset:
                return opt["note"]
        return "Any Hugging Face model id (org/name) or a local model folder."

    @rx.var
    def method_data_hint(self) -> str:
        return METHOD_DATA_HINTS.get(self.method, METHOD_DATA_HINTS["sft"])

    @rx.var
    def is_full_ft(self) -> bool:
        return self.train_mode == "full"

    @rx.var
    def vram_est_title(self) -> str:
        return VRAM_VERDICT_TITLES.get(self.vram_est_verdict, "No estimate")

    @rx.var
    def vram_est_label(self) -> str:
        """"15.4 GB of 31.8 GB" (both GiB, like `backprop estimate-vram`)."""
        if self.vram_est_total <= 0:
            return ""
        if self.vram_total_gb > 0:
            return f"{self.vram_est_total:.1f} GB of {self.vram_total_gb:.1f} GB"
        return f"{self.vram_est_total:.1f} GB"

    @rx.var
    def vram_est_pct(self) -> str:
        if self.vram_est_total <= 0 or self.vram_total_gb <= 0:
            return "0%"
        return f"{min(100.0, 100.0 * self.vram_est_total / self.vram_total_gb):.1f}%"

    @rx.var
    def vram_est_detail(self) -> str:
        if self.vram_est_note:
            return self.vram_est_note
        lead = (
            "Measured on this GPU for"
            if self.vram_est_source == "measured"
            else "An estimate for"
        )
        batch = (
            f"batch auto picks {self.vram_est_batch} on this card"
            if self.batch_size == "auto" and self.vram_est_batch
            else f"batch {self.vram_est_batch}"
        )
        advice = {
            "fits": "",
            "tight": " Close to the limit: lower the batch or the LoRA rank if it runs out of memory.",
            "wont_fit": " Try a lower batch size first, then QLoRA, a lower LoRA rank or a smaller model.",
        }.get(self.vram_est_verdict, "")
        return f"{lead} {batch} with full 2,048-token rows; shorter data uses less.{advice}"

    @rx.var
    def vram_est_heading(self) -> str:
        return (
            "VRAM · measured on this GPU"
            if self.vram_est_source == "measured"
            else "VRAM · estimate"
        )

    @rx.var
    def vram_est_measured(self) -> bool:
        return self.vram_est_source == "measured"

    @rx.var
    def vram_fill_pct(self) -> str:
        """Live width string for the VRAM bar (e.g. ``"29.7%"``)."""
        if not self.vram_total_gb or self.vram_total_gb <= 0:
            return "0.0%"
        ratio = max(0.0, min(1.0, self.vram_used_gb / self.vram_total_gb))
        return f"{ratio * 100:.1f}%"

    @rx.var
    def vram_label(self) -> str:
        """``"9.5 / 32.0 GB"`` for the VRAM bar label."""
        if not self.vram_total_gb or self.vram_total_gb <= 0:
            return "VRAM unknown"
        return f"{self.vram_used_gb:.1f} / {self.vram_total_gb:.1f} GB"

    @rx.var
    def gpu_temp_label(self) -> str:
        """``"61°C"`` — empty when no reading exists (honest idle)."""
        if not self.gpu_temp or self.gpu_temp <= 0:
            return ""
        return f"{self.gpu_temp:.0f}°C"

    @rx.var
    def gpu_fill_pct(self) -> str:
        """``"64%"`` arc fill for the temp ring (clamped to 95°C right edge)."""
        if not self.gpu_temp or self.gpu_temp <= 0:
            return "0%"
        ratio = max(0.0, min(1.0, self.gpu_temp / 95.0))
        return f"{ratio * 100:.0f}%"

    @rx.var
    def step_progress_pct(self) -> str:
        """Banner progress width: current_step / job_total_steps."""
        if not self.job_total_steps or self.job_total_steps <= 0:
            return "0.0%"
        ratio = max(0.0, min(1.0, self.current_step / self.job_total_steps))
        return f"{ratio * 100:.1f}%"

    @rx.var
    def run_complete(self) -> bool:
        """True when the run has reached a terminal state (``done`` or ``error``).

        FRONTEND-10 (post-run "next steps" panel) reads this Var to know when
        to surface the post-run affordances. ``error`` counts as complete for
        UI purposes — the operator still wants the "export what you have /
        start another / view checkpoints" affordances after a failure. A
        cooperative stop (``stopped``) is complete too: the checkpoint was
        saved, so the same affordances apply.
        """
        return self.run_state in ("done", "stopped", "error")

    # ---- Recovery-banner Vars (FRONTEND-A-004, v1.4 Wave 2) -----------------
    #
    # The Train page surfaces the MOST RECENT ``ok`` / ``warn`` event as a
    # ``BpRecoveryBanner``. Pre-fix the component existed but no page rendered
    # it (the docstring at pages/train.py:6 even claimed "Recovery banners
    # (when applicable)" but the body never wired one). The component takes
    # plain Python strings for the variant (color + icon are looked up at
    # component-build time, not via Vars), so the three variants are exposed
    # as three separate Vars and the page renders three conditional banners.
    #
    # "Most recent" means walking the events list in reverse and returning
    # the first ok / warn entry; if none, return an empty string and the
    # page's ``rx.cond`` keeps the banner unmounted.

    @rx.var
    def latest_recovery_ok_msg(self) -> str:
        """Message of the most recent ``ok``-level event, or empty.

        Drives the Train page's "good news" recovery banner — e.g. trainer
        successfully resumed from an OOM bisect, GPU temp dropped below
        threshold, checkpoint was written after a near-miss.
        """
        for ev in reversed(self.events):
            if isinstance(ev, dict) and ev.get("level") == "ok":
                return str(ev.get("msg") or "")
        return ""

    @rx.var
    def latest_recovery_warn_msg(self) -> str:
        """Message of the most recent ``warn``-level event, or empty.

        Drives the Train page's "heads-up" recovery banner — e.g. GPU temp
        approaching threshold, dataset row skipped, batch auto-shrunk for
        VRAM. Distinct from ``err``: warn is a recovered-or-recoverable
        condition; err is a hard failure. HUX-01: ``err``-level events are
        NOT surfaced on the Train page today — UI training is an honest stub
        (``start_training`` directs the operator to ``backprop train``), so no
        ``BpErrorCallout`` is wired on this page yet. The structured callout
        is consumed by the /runs, /models, and /run-detail pages.
        """
        for ev in reversed(self.events):
            if isinstance(ev, dict) and ev.get("level") == "warn":
                return str(ev.get("msg") or "")
        return ""

    @rx.var
    def latest_recovery_info_msg(self) -> str:
        """Message of the most recent ``info``-level event, or empty.

        Drives the Train page's neutral-tint banner — e.g. "trainer started",
        "dataset loaded", "exporting adapter". This one fires for routine
        lifecycle events; the page renders it at lower visual weight than
        ok / warn (or the page may opt to hide it entirely — see the
        train.py wiring rationale).
        """
        for ev in reversed(self.events):
            if isinstance(ev, dict) and ev.get("level") == "info":
                return str(ev.get("msg") or "")
        return ""

    # ---- Setters (FRONTEND-A-002 + FRONTEND-B-002) -------------------------

    @rx.event
    def set_model(self, value: str):
        self.model, self.model_error = _apply_model(value)
        self.preset = _preset_for_model(self.model)
        return TrainState.refresh_estimate

    @rx.event
    def set_preset(self, key: str):
        """Fill the model and its recommended LoRA rank (alpha = 2 x rank)."""
        for opt in model_preset_options():
            if opt["key"] == key:
                self.preset = key
                self.model, self.model_error = opt["model_id"], ""
                self.lora_r = int(opt["lora_r"])
                self.lora_alpha = 2 * self.lora_r
                self.lora_r_error = self.lora_alpha_error = ""
                return TrainState.refresh_estimate
        self.preset = "custom"

    @rx.event
    def set_train_mode(self, value: str):
        if value not in ("qlora", "lora", "full"):
            return
        if value == "full" and self.method != "sft":
            self.job_refusal = (
                f"{self.method.upper()} trains a LoRA adapter; switch the method "
                "to SFT for full fine-tuning."
            )
            return
        self.train_mode = value  # type: ignore[assignment]
        return TrainState.refresh_estimate

    @rx.event
    def set_method(self, value: str) -> None:
        if value not in _METHOD_KEYS:
            return
        self.method = value  # type: ignore[assignment]
        self.method_param_error = ""
        if value != "sft" and self.train_mode == "full":
            self.train_mode = "qlora"

    @rx.event
    def set_method_param(self, key: str, value: str | float) -> None:
        if key not in METHOD_DEFAULTS:
            return
        f, err = _clamp_float(key.replace("_", " "), value, 1e-6, _METHOD_PARAM_MAX)
        if f is not None:
            setattr(self, key, f)
        self.method_param_error = err

    @rx.event
    def apply_lora_shape(self, shape: str):
        if shape not in LORA_SHAPES:
            return
        self.lora_r, self.lora_alpha, self.target_modules = LORA_SHAPES[shape]
        self.lora_r_error = self.lora_alpha_error = self.target_modules_error = ""
        return TrainState.refresh_estimate

    @rx.event
    def set_target_modules(self, value: str):
        new, err = _apply_target_modules(value)
        if new is not None:
            self.target_modules = new
        self.target_modules_error = err
        return TrainState.refresh_estimate

    @rx.event
    def set_dataset_path(self, value: str) -> None:
        self.dataset_path, self.dataset_path_error = _validate_ui_path(value)

    @rx.event
    def set_steps(self, value: str | int) -> None:
        n, err = _clamp_int("Steps", value, _STEPS_MIN, _STEPS_MAX)
        if n is not None:
            self.steps = n
        self.steps_error = err

    @rx.event
    def set_batch_size(self, value: str):
        new, err = _apply_batch_size(value)
        if new is not None:
            self.batch_size = new
        self.batch_size_error = err
        return TrainState.refresh_estimate

    @rx.event
    def set_learning_rate(self, value: str | float) -> None:
        f, err = _clamp_float("Learning rate", value, _LR_MIN, _LR_MAX)
        if f is not None:
            self.learning_rate = f
        self.learning_rate_error = err

    @rx.event
    def set_lora_r(self, value: str | int):
        n, err = _clamp_int("LoRA rank", value, _LORA_R_MIN, _LORA_R_MAX)
        if n is not None:
            self.lora_r = n
        self.lora_r_error = err
        return TrainState.refresh_estimate

    @rx.event(background=True)
    async def refresh_estimate(self):
        """Recompute the inline VRAM estimate off the event loop (the
        estimator's first import loads the trainer module)."""
        import asyncio

        from .ui_jobs import vram_verdict

        async with self:
            self._vram_est_seq += 1
            seq = self._vram_est_seq
            args = (
                self.model,
                "full" if self.train_mode == "full" else "lora",
                int(self.lora_r),
                str(self.batch_size),
                self.train_mode != "lora",
                bool(self.gradient_checkpointing),
                float(self.vram_total_gb or 0.0),
                str(self.target_modules).replace(" ", ""),
            )
        result = await asyncio.to_thread(
            lambda: vram_verdict(
                args[0], mode=args[1], lora_r=args[2], batch=args[3],
                base_4bit=args[4], gradient_checkpointing=args[5], card_gb=args[6],
                target_modules=args[7],
            )
        )
        async with self:
            if seq != self._vram_est_seq:
                return  # a newer refresh started; it owns the numbers
            self.vram_est_verdict = str(result.get("verdict") or "unknown")
            self.vram_est_total = float(result.get("total_gb") or 0.0)
            self.vram_est_batch = int(result.get("batch") or 0)
            self.vram_est_note = str(result.get("note") or "")
            self.vram_est_source = str(result.get("source") or "estimate")

    def _estimate_args(self) -> dict:
        return {
            "mode": "full" if self.train_mode == "full" else "lora",
            "lora_r": int(self.lora_r),
            "batch": str(self.batch_size),
            "base_4bit": self.train_mode != "lora",
            "gradient_checkpointing": bool(self.gradient_checkpointing),
            "card_gb": float(self.vram_total_gb or 0.0),
            "target_modules": str(self.target_modules).replace(" ", ""),
        }

    def _refresh_estimate_now(self) -> None:
        """Recompute the estimate in place (after a calibration finishes)."""
        from .ui_jobs import vram_verdict

        result = vram_verdict(self.model, **self._estimate_args())
        self._vram_est_seq += 1
        self.vram_est_verdict = str(result.get("verdict") or "unknown")
        self.vram_est_total = float(result.get("total_gb") or 0.0)
        self.vram_est_batch = int(result.get("batch") or 0)
        self.vram_est_note = str(result.get("note") or "")
        self.vram_est_source = str(result.get("source") or "estimate")

    @rx.event
    def start_calibration(self):
        """Measure this model on this GPU (``estimate-vram --calibrate``) as a
        job: one at a time, with the log, the phase and Stop like any other."""
        from .ui_jobs import JobSpec

        self.job_refusal = ""
        if self.model_error:
            self.job_refusal = "Fix the model field first: " + self.model_error
            return
        spec = JobSpec(
            kind="calibrate",
            model=self.model,
            mode="full" if self.train_mode == "full" else "lora",
            base_4bit=self.train_mode != "lora",
        )
        return self._begin_job(spec)

    @rx.event
    def set_lora_alpha(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA alpha", value, _LORA_ALPHA_MIN, _LORA_ALPHA_MAX)
        if n is not None:
            self.lora_alpha = n
        self.lora_alpha_error = err

    @rx.event
    def set_lora_dropout(self, value: str | float) -> None:
        f, err = _clamp_float(
            "Dropout", value, _LORA_DROPOUT_MIN, _LORA_DROPOUT_MAX
        )
        if f is not None:
            self.lora_dropout = f
        self.lora_dropout_error = err

    @rx.event
    def set_gpu_temp_threshold(self, value: str | int) -> None:
        n, err = _clamp_int(
            "GPU temp threshold", value, _GPU_TEMP_MIN, _GPU_TEMP_MAX
        )
        if n is not None:
            self.gpu_temp_threshold = n
        self.gpu_temp_threshold_error = err

    @rx.event
    def set_wandb_run_name(self, value: str) -> None:
        new, err = _apply_wandb_run_name(value)
        if new is not None:
            self.wandb_run_name = new
        self.wandb_run_name_error = err

    @rx.event
    def set_gradient_checkpointing(self, value: bool):
        self.gradient_checkpointing = bool(value)
        return TrainState.refresh_estimate

    # ---- Live-run plumbing (ui-v2 P1) -----------------------------------------
    # Populated by the background poller from the job's events.jsonl. The job
    # itself is a subprocess spawned by backpropagate.ui_jobs.JobManager —
    # training NEVER runs in the UI server process.
    job_id: str = ""
    job_phase: str = ""
    job_total_steps: int = 0
    job_refusal: str = ""
    job_error_code: str = ""
    job_error_message: str = ""
    job_error_hint: str = ""
    job_output_path: str = ""
    job_stalled: bool = False
    # ui-v2 P2: TrainState follows EVERY UI job (one at a time), so the
    # multi-run and export pages share the progress card, rail and reattach.
    job_kind: str = "sft"  # "sft" | "multi_run" | "export"
    job_run: int = 0  # multi-run: current run (1-based)
    job_runs: int = 0  # multi-run: total runs
    # Requirement 10: on failure, the last log lines next to the error code.
    job_log_tail: list[str] = []
    gpu_name: str = ""
    # ui-v2 P1 fix round: set ONLY when this page opened onto a run it did not
    # start (adopted mid-flight). Drives the single "Reattached…" banner — the
    # old latest-ok/latest-warn banners fired "Recovered." after every normal
    # finish, which was wrong framing.
    reattach_notice: str = ""
    _job_offset: int = 0
    _last_step_epoch: float = 0.0
    _stop_deadline: float = 0.0
    _stop_pending: bool = False
    _poll_running: bool = False
    _step_time_ms_ema: float = 0.0
    _step_samples: int = 0

    # ---- Event handlers (stubs; backend hookup in Phase 3) -----------------

    @rx.event
    def refuse(self, message: str) -> None:
        """Show a start refusal from another page's form (ui-v2 P2)."""
        self.job_refusal = str(message)

    @rx.event
    def dismiss_refusal(self) -> None:
        """Clear the start-refusal banner (ui-v2 P1)."""
        self.job_refusal = ""

    @rx.event
    def refresh_gpu(self) -> None:
        """Fill the side rail's GPU block from this machine's live readings.

        Idle state shows DEVICE-WIDE usage (via pynvml → nvidia-smi): the
        operator wants "the card has 2.4 GB in use", not the UI server's
        own ~0 GB slice. During a run the step events (written by the CHILD
        process) overwrite these fields with the training view.
        """
        try:
            from .gpu_safety import get_gpu_status, get_system_gpu_readings

            readings = get_system_gpu_readings()
            status = get_gpu_status()
            if readings is not None and readings.device_name:
                self.gpu_name = readings.device_name
            elif status.available:
                self.gpu_name = status.device_name
            else:
                self.gpu_name = "No CUDA GPU"
            if not self.job_id or self.run_state != "active":
                if readings is not None:
                    if readings.memory_total_gib > 0:
                        self.vram_total_gb = round(readings.memory_total_gib, 1)
                    self.vram_used_gb = round(readings.memory_used_gib, 2)
                    if readings.temperature_c is not None:
                        self.gpu_temp = float(readings.temperature_c)
                else:
                    if status.vram_total_gb and status.vram_total_gb > 0:
                        self.vram_total_gb = round(float(status.vram_total_gb), 1)
                    if status.vram_used_gb is not None:
                        self.vram_used_gb = round(float(status.vram_used_gb), 2)
                    if status.temperature_c is not None:
                        self.gpu_temp = float(status.temperature_c)
        except Exception:  # noqa: BLE001 — telemetry must never break the page
            if not self.gpu_name:
                self.gpu_name = "GPU status unavailable"

    @rx.event
    def start_training(self):
        """Start an SFT run in a child process (ui-v2 P1).

        Validation failures land in the refusal callout on screen (never a
        fake spinner); a second concurrent start is refused the same way.
        """
        from .ui_jobs import JobSpec

        self.job_refusal = ""
        form_errors = _form_errors(self)
        if form_errors:
            self.job_refusal = "Fix the highlighted fields first: " + "; ".join(form_errors)
            return
        # ui-v2 P3: every field maps to a real `backprop train` flag
        # (ui_jobs._training_flags); the form shows the CLI's defaults.
        spec = JobSpec(kind="sft", **_training_spec_fields(self))
        return self._begin_job(spec)

    @rx.event
    def start_job(self, payload: dict):
        """Start a multi-run or export job for the other pages (ui-v2 P2).

        ``payload`` holds JobSpec fields; MultiRunState / ExportState
        validate their forms, then hand off here so every job shares one
        poller, one progress card and one reattach path.
        """
        from .ui_jobs import SERVER_ONLY_SPEC_FIELDS, JobSpec

        self.job_refusal = ""
        # This handler is reachable from the browser, so ``payload`` is
        # untrusted: a client may set form fields only, never where the job
        # writes (output_dir / scratch_root) or trust_remote_code.
        allowed = set(JobSpec.__dataclass_fields__) - SERVER_ONLY_SPEC_FIELDS
        spec_kwargs = {k: v for k, v in dict(payload).items() if k in allowed}
        try:
            spec = JobSpec(**spec_kwargs)
        except TypeError as exc:
            self.job_refusal = f"Could not start: {exc}"
            return None
        return self._begin_job(spec)

    def _begin_job(self, spec):
        """Spawn ``spec`` through the JobManager and reset the live state.

        Returns the poller event, or None when the start was refused (the
        reason lands in ``job_refusal``, on screen).
        """
        from .ui_jobs import JobRefusedError, JobValidationError, get_job_manager

        try:
            handle = get_job_manager().start(spec)
        except (JobValidationError, JobRefusedError, NotImplementedError) as exc:
            self.job_refusal = str(exc)
            return None
        except Exception as exc:  # noqa: BLE001 — spawn failure lands on screen
            self.job_refusal = f"Could not start: {exc}"
            return None
        self.job_id = handle.job_id
        self.job_kind = spec.kind
        self.job_run = 0
        self.job_runs = int(spec.runs) if spec.kind == "multi_run" else 0
        self.job_phase = "queued"
        if spec.kind == "multi_run":
            self.job_total_steps = int(spec.runs) * int(spec.steps)
        elif spec.kind in ("export", "calibrate"):
            self.job_total_steps = 0
        else:
            self.job_total_steps = int(spec.steps)
        self.job_error_code = ""
        self.job_error_message = ""
        self.job_error_hint = ""
        self.job_output_path = ""
        self.job_log_tail = []
        self.job_safety_reason = ""
        self.job_stalled = False
        self.run_state = "active"
        self.current_step = 0
        self.current_loss = 0.0
        self.loss_history = []
        self.ema_history = []
        self.step_history = []
        self.stop_requested = False
        self.reattach_notice = ""
        self.heartbeat_label = ""
        self.eta_label = ""
        self._step_time_ms_ema = 0.0
        self._step_samples = 0
        self._job_offset = 0
        import time as _time

        self._last_step_epoch = _time.time()
        what = {
            "multi_run": "Multi-run",
            "export": "Export",
            "calibrate": "Measurement",
        }.get(spec.kind, "Run")
        self.events = [
            *self.events,
            {
                "t": _ts_now(),
                "level": "info",
                "msg": f"{what} {handle.job_id} started in a separate process.",
            },
        ]
        return TrainState.poll_job

    @rx.event
    def attach_active_job(self):
        """Adopt/finalize on mount (ui-v2 P1 fix round).

        Two cases:
        - This page already knows the job (same session after a reload): the
          poller may have died with the old socket, so immediately check for
          a terminal state on disk (fixes 'reload right after the end still
          shows active + Stop'); if still live and no poller is running,
          restart it.
        - A fresh session opens mid-run: adopt the live job so the page
          shows REAL progress instead of pretending idle. This is the ONLY
          path that sets ``reattach_notice`` — the banner means 'this page
          did not start this run', never 'your run finished'.
        """
        from .ui_jobs import get_job_manager

        manager = get_job_manager()
        if self.job_id:
            if self.run_state == "active":
                status = manager.status()
                if status.get("status") in ("done", "stopped", "failed", "crashed"):
                    self._finalize_job(status)
                    return
                if not self._poll_running:
                    return TrainState.poll_job
            return
        status = manager.status()
        jid = str(status.get("job_id") or "")
        if status.get("status") != "active" or not jid:
            return
        # Adopt: replays events.jsonl from byte 0 so the chart/phase rebuild.
        self.job_id = jid
        self.job_kind = str(status.get("kind") or "sft")
        self.run_state = "active"
        self.job_phase = str(status.get("phase") or "training")
        self.job_total_steps = int(status.get("total_steps") or 0)
        self.current_step = int(status.get("step") or 0)
        self._job_offset = 0
        import time as _time

        self._last_step_epoch = _time.time()
        self.reattach_notice = (
            f"Reattached to a run this page didn't start ({jid}). "
            "Progress below is live; Stop and save works the same."
        )
        self.events = [
            *self.events,
            {"t": _ts_now(), "level": "info", "msg": self.reattach_notice},
        ]
        if not self._poll_running:
            return TrainState.poll_job

    @rx.event(background=True)
    async def poll_job(self) -> None:
        """Tail events.jsonl ~1/s and mirror it into the state (ui-v2 P1)."""
        import asyncio
        import time as _time

        from .ui_jobs import get_job_manager

        manager = get_job_manager()
        async with self:
            if self._poll_running:
                return  # a second poller would race the byte offset
            self._poll_running = True
        try:
            while True:
                async with self:
                    if not self.job_id:
                        break
                    rows, offset = manager.tail_events(self.job_id, self._job_offset)
                    self._job_offset = offset
                    for row in rows:
                        if row.get("kind") == "step":
                            self._last_step_epoch = _time.time()
                            self.job_stalled = False
                        self._apply_job_event(row)
                # heartbeat: no step event for 120 s while active -> warn once
                async with self:
                    self._tick_live_labels()
                    if (
                        self.run_state == "active"
                        and self.job_phase == "training"
                        and not self.job_stalled
                        and _time.time() - self._last_step_epoch > 120
                    ):
                        self.job_stalled = True
                        self.events = [
                            *self.events,
                            {
                                "t": _ts_now(),
                                "level": "warn",
                                "msg": "No training progress for 2 minutes — "
                                "check GPU activity before assuming a wedge.",
                            },
                        ]
                    status = manager.status()
                    if status.get("status") in ("done", "stopped", "failed", "crashed"):
                        self._finalize_job(status)
                        break
                    # Stop escalation: grace window expired and the child still
                    # breathes -> tree kill (ui-v2 P1 cooperative-stop contract).
                    if (
                        self._stop_pending
                        and _time.monotonic() > self._stop_deadline
                        and manager.is_alive(self.job_id)
                    ):
                        self._stop_pending = False
                        manager.hard_kill(self.job_id)
                        self.events = [
                            *self.events,
                            {
                                "t": _ts_now(),
                                "level": "warn",
                                "msg": "Grace expired — training tree force-killed.",
                            },
                        ]
                await asyncio.sleep(1.0)
        finally:
            async with self:
                self._poll_running = False

    def _apply_job_event(self, row: dict) -> None:
        """Fold one events.jsonl row into the visible state. Plain method —
        called from the poller inside ``async with self``."""
        kind = row.get("kind")
        if kind == "phase":
            phase = str(row.get("phase") or "")
            if phase and phase != self.job_phase:
                self.job_phase = phase
                level = "ok" if phase == "done" else "info"
                self.events = [
                    *self.events,
                    {"t": _ts_now(), "level": level, "msg": f"Phase: {phase}"},
                ]
        elif kind == "step":
            self.current_step = int(row.get("step") or 0)
            total = row.get("total_steps") or 0
            if total:
                self.job_total_steps = int(total)
            loss = row.get("loss")
            if isinstance(loss, (int, float)):
                self.current_loss = float(loss)
                self.loss_history = [*self.loss_history[-79:], float(loss)]
                self.step_history = [
                    *self.step_history[-79:],
                    self.current_step,
                ]
                ema = row.get("ema_loss")
                self.ema_history = [
                    *self.ema_history[-79:],
                    float(ema) if isinstance(ema, (int, float)) else float(loss),
                ]
            stp = row.get("step_time_ms")
            if isinstance(stp, (int, float)) and stp > 0:
                # EMA (alpha 0.35) of the per-step wall time feeds the ETA
                # range; warm-up gate: need a few samples before quoting it.
                if self._step_samples == 0:
                    self._step_time_ms_ema = float(stp)
                else:
                    self._step_time_ms_ema = (
                        0.35 * float(stp) + 0.65 * self._step_time_ms_ema
                    )
                self._step_samples += 1
            vram = (
                row.get("vram_device_used_gib")
                or row.get("vram_reserved_gib")
                or row.get("vram_alloc_gib")
            )
            if isinstance(vram, (int, float)):
                self.vram_used_gb = round(float(vram), 2)
            vram_total = row.get("vram_device_total_gib")
            if isinstance(vram_total, (int, float)) and vram_total > 0:
                self.vram_total_gb = round(float(vram_total), 1)
            temp = row.get("temp_c")
            if isinstance(temp, (int, float)) and temp:
                self.gpu_temp = float(temp)
        elif kind == "safety":
            # --gpu-max-temp tripped: the child saves and stops (same path as
            # the Stop button). Shown in the feed and in the stopped label.
            self.job_safety_reason = str(row.get("reason") or "GPU temperature limit reached")
            self.events = [
                *self.events,
                {"t": _ts_now(), "level": "warn", "msg": self.job_safety_reason},
            ]
        elif kind == "run":
            self.job_run = int(row.get("run") or 0)
            self.job_runs = int(row.get("runs") or self.job_runs or 0)
            self.events = [
                *self.events,
                {
                    "t": _ts_now(),
                    "level": "info",
                    "msg": f"Run {self.job_run} of {self.job_runs} started.",
                },
            ]
        elif kind == "checkpoint":
            self.events = [
                *self.events,
                {
                    "t": _ts_now(),
                    "level": "ok",
                    # The folder name only: a full path wrapped to ~8 lines
                    # in the 296px rail. "Saved to:" shows the full path.
                    "msg": "Checkpoint saved: "
                    + (Path(str(row.get("path") or "")).name or str(row.get("path") or "")),
                },
            ]
        elif kind == "error":
            self.job_error_code = str(row.get("code") or "RUNTIME_ERROR")
            self.job_error_message = str(row.get("message") or "")
            self.job_error_hint = str(row.get("hint") or "")
            self.events = [
                *self.events,
                {
                    "t": _ts_now(),
                    "level": "err",
                    "msg": f"{self.job_error_code}: {self.job_error_message[:200]}",
                },
            ]

    def _tick_live_labels(self) -> None:
        """Refresh the heartbeat + ETA labels (called once per poll tick).

        Requirement 10 of the ui-v2 digest: the operator should always see
        'last step N s ago' (staleness at a glance) and, after warm-up, a
        RANGE ETA ('about 3–5 min left') derived from the step-time EMA —
        a range stays honest when steps vary (compile, eval, save).
        """
        import time as _time

        if self.run_state != "active":
            return
        ago = max(0, int(_time.time() - self._last_step_epoch))
        self.heartbeat_label = (
            f"last step {ago} s ago" if self.job_phase == "training" else ""
        )
        # Warm-up per the handoff: about 20 steps or 5% of the run, whichever
        # comes first, and at least two step-time samples for the EMA (steps
        # are logged every 10, so a 5-sample floor hid the ETA until step 50).
        warm_step = min(20, max(1, int(self.job_total_steps * 0.05)))
        if (
            self._step_samples >= 2
            and self.current_step >= warm_step
            and self.job_total_steps > self.current_step > 0
            and self._step_time_ms_ema > 0
        ):
            remaining = self.job_total_steps - self.current_step
            eta_s = self._step_time_ms_ema * remaining / 1000.0
            lo, hi = _fmt_eta_range(eta_s)
            self.eta_label = f"about {lo}–{hi} left" if lo != hi else f"about {lo} left"
        else:
            self.eta_label = ""

    def _finalize_job(self, status: dict) -> None:
        """Terminal bookkeeping shared by the poller and the stopper."""
        outcome = str(status.get("status") or "done")
        if outcome in ("failed", "crashed"):
            self.run_state = "error"
        elif outcome == "stopped":
            # Honest state: the rail chip + page read "stopped", not "done".
            self.run_state = "stopped"
        else:
            self.run_state = "done"
        self.stop_requested = False
        self._stop_pending = False
        self.heartbeat_label = ""
        self.eta_label = ""
        out_path = str(status.get("output_path") or "")
        if out_path:
            self.job_output_path = out_path
        if outcome in ("failed", "crashed"):
            self.job_log_tail = _read_log_tail(str(status.get("log_path") or ""))
        if self.job_kind == "calibrate":
            labels = {
                "done": "Measured. The estimate now uses this GPU's own numbers.",
                "stopped": "Measurement cancelled.",
                "failed": "The measurement could not be completed.",
                "crashed": "The measurement process died unexpectedly (crashed).",
            }
            if outcome == "done":
                self._refresh_estimate_now()
        elif self.job_kind == "export":
            labels = {
                "done": "Export finished.",
                "stopped": "Export cancelled.",
                "failed": "Export failed.",
                "crashed": "The export process died unexpectedly (crashed).",
            }
        elif self.job_kind == "multi_run":
            labels = {
                "done": "Multi-run completed.",
                "stopped": "Stopped early — the runs merged so far were kept.",
                "failed": "Multi-run failed.",
                "crashed": "The multi-run process died unexpectedly (crashed).",
            }
        else:
            labels = {
                "done": "Run completed.",
                "stopped": "Stopped early — the checkpoint was saved before the halt.",
                "failed": "Run failed.",
                "crashed": "The training process died unexpectedly (crashed).",
            }
        label = labels.get(outcome, outcome)
        if (
            outcome == "stopped"
            and self.job_safety_reason
            and self.job_kind not in ("export", "calibrate")
        ):
            label = (
                f"Stopped by the GPU temperature limit ({self.job_safety_reason}). "
                "The checkpoint was saved."
            )
        self.events = [
            *self.events,
            {
                "t": _ts_now(),
                "level": "ok" if outcome == "done" else ("warn" if outcome == "stopped" else "err"),
                "msg": label,
            },
        ]

    @rx.event
    def stop_training(self) -> None:
        """Stop and save checkpoint (ui-v2 P1).

        Writes control.json (the child saves at the next step boundary) and
        arms the escalation deadline; the background poller performs the
        tree kill if the grace window expires. Kept as a FOREGROUND event so
        it stays unit-testable and never blocks on the grace wait.
        """
        import time as _time

        from .ui_jobs import get_job_manager

        manager = get_job_manager()
        if (
            self.job_kind in ("export", "calibrate")
            and self.job_id
            and manager.is_alive(self.job_id)
        ):
            # No step boundary to save at: cancel kills the process tree now
            # and records the outcome as stopped.
            manager.cancel(self.job_id)
            self.stop_requested = True
            what = "export" if self.job_kind == "export" else "measurement"
            self.events = [
                *self.events,
                {"t": _ts_now(), "level": "warn", "msg": f"Cancelling the {what}."},
            ]
            return
        if self.job_id and manager.is_alive(self.job_id):
            grace = manager.grace_window(self.job_id)
            manager.request_stop(self.job_id)
            self._stop_deadline = _time.monotonic() + grace
            self._stop_pending = True
            self.stop_requested = True
            self.events = [
                *self.events,
                {
                    "t": _ts_now(),
                    "level": "warn",
                    "msg": f"Stop requested — saving a checkpoint "
                    f"(grace {int(grace)}s before a hard stop).",
                },
            ]
        elif self.run_state in ("active", "loading", "paused"):
            self.run_state = "idle"
            self.stop_requested = False


class MultiRunState(rx.State):
    """Multi-Run surface state: config + num_runs, samples_per_run,
    merge_mode, replay_fraction.

    Config fields duplicate TrainState's by design (see TrainState docstring);
    setter logic routes through the same module-level helpers.
    """

    # ---- Configuration form (mirrors TrainState; CLI defaults) -------------
    preset: str = "qwen2.5-7b"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    model_error: str = ""
    dataset_path: str = ""
    dataset_path_error: str = ""
    steps: int = 100
    steps_error: str = ""
    batch_size: str = "auto"
    batch_size_error: str = ""
    learning_rate: float = 2e-4
    learning_rate_error: str = ""
    lora_r: int = 256
    lora_r_error: str = ""
    lora_alpha: int = 512
    lora_alpha_error: str = ""
    lora_dropout: float = 0.05
    lora_dropout_error: str = ""
    target_modules: str = "all-linear"
    target_modules_error: str = ""
    train_mode: TrainMode = "qlora"
    method: Method = "sft"
    orpo_beta: float = 0.1
    simpo_beta: float = 2.0
    simpo_gamma: float = 1.0
    kto_beta: float = 0.1
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
    method_param_error: str = ""
    gpu_temp_threshold: int = 90
    gpu_temp_threshold_error: str = ""
    wandb_run_name: str = ""
    wandb_run_name_error: str = ""
    gradient_checkpointing: bool = True

    # ---- Multi-Run specific ------------------------------------------------
    num_runs: int = 3
    num_runs_error: str = ""
    samples_per_run: int = 500
    samples_per_run_error: str = ""
    merge_mode: MergeMode = "slao"
    replay_fraction: float = 0.0
    replay_fraction_error: str = ""

    # ---- Live state --------------------------------------------------------
    run_state: RunState = "idle"
    current_run_index: int = 0
    runs: list[dict] = []  # per-run summary (loss, step, status)
    events: list[dict] = []

    # ---- Setters (shared logic with TrainState via _apply_* helpers) -------

    @rx.event
    def set_model(self, value: str) -> None:
        self.model, self.model_error = _apply_model(value)
        self.preset = _preset_for_model(self.model)

    @rx.event
    def set_preset(self, key: str) -> None:
        """Fill the model and its recommended LoRA rank (alpha = 2 x rank)."""
        for opt in model_preset_options():
            if opt["key"] == key:
                self.preset = key
                self.model, self.model_error = opt["model_id"], ""
                self.lora_r = int(opt["lora_r"])
                self.lora_alpha = 2 * self.lora_r
                self.lora_r_error = self.lora_alpha_error = ""
                return
        self.preset = "custom"

    @rx.event
    def set_train_mode(self, value: str) -> None:
        # A multi-run merges LoRA adapters: QLoRA or LoRA, never full.
        if value in ("qlora", "lora"):
            self.train_mode = value  # type: ignore[assignment]

    @rx.event
    def set_method(self, value: str) -> None:
        if value in _METHOD_KEYS:
            self.method = value  # type: ignore[assignment]
            self.method_param_error = ""

    @rx.event
    def set_method_param(self, key: str, value: str | float) -> None:
        if key not in METHOD_DEFAULTS:
            return
        f, err = _clamp_float(key.replace("_", " "), value, 1e-6, _METHOD_PARAM_MAX)
        if f is not None:
            setattr(self, key, f)
        self.method_param_error = err

    @rx.event
    def apply_lora_shape(self, shape: str) -> None:
        if shape in LORA_SHAPES:
            self.lora_r, self.lora_alpha, self.target_modules = LORA_SHAPES[shape]
            self.lora_r_error = self.lora_alpha_error = self.target_modules_error = ""

    @rx.var
    def lora_shape(self) -> str:
        return _lora_shape_of(self.lora_r, self.lora_alpha, self.target_modules)

    @rx.var
    def preset_note(self) -> str:
        for opt in model_preset_options():
            if opt["key"] == self.preset:
                return opt["note"]
        return "Any Hugging Face model id (org/name) or a local model folder."

    @rx.var
    def method_data_hint(self) -> str:
        return METHOD_DATA_HINTS.get(self.method, METHOD_DATA_HINTS["sft"])

    @rx.var
    def is_full_ft(self) -> bool:
        return False

    @rx.event
    def set_dataset_path(self, value: str) -> None:
        self.dataset_path, self.dataset_path_error = _validate_ui_path(value)

    @rx.event
    def set_steps(self, value: str | int) -> None:
        n, err = _clamp_int("Steps", value, _STEPS_MIN, _STEPS_MAX)
        if n is not None:
            self.steps = n
        self.steps_error = err

    @rx.event
    def set_batch_size(self, value: str) -> None:
        new, err = _apply_batch_size(value)
        if new is not None:
            self.batch_size = new
        self.batch_size_error = err

    @rx.event
    def set_learning_rate(self, value: str | float) -> None:
        f, err = _clamp_float("Learning rate", value, _LR_MIN, _LR_MAX)
        if f is not None:
            self.learning_rate = f
        self.learning_rate_error = err

    @rx.event
    def set_lora_r(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA rank", value, _LORA_R_MIN, _LORA_R_MAX)
        if n is not None:
            self.lora_r = n
        self.lora_r_error = err

    @rx.event
    def set_lora_alpha(self, value: str | int) -> None:
        n, err = _clamp_int("LoRA alpha", value, _LORA_ALPHA_MIN, _LORA_ALPHA_MAX)
        if n is not None:
            self.lora_alpha = n
        self.lora_alpha_error = err

    @rx.event
    def set_lora_dropout(self, value: str | float) -> None:
        f, err = _clamp_float(
            "Dropout", value, _LORA_DROPOUT_MIN, _LORA_DROPOUT_MAX
        )
        if f is not None:
            self.lora_dropout = f
        self.lora_dropout_error = err

    @rx.event
    def set_target_modules(self, value: str) -> None:
        new, err = _apply_target_modules(value)
        if new is not None:
            self.target_modules = new
        self.target_modules_error = err

    @rx.event
    def set_gpu_temp_threshold(self, value: str | int) -> None:
        n, err = _clamp_int(
            "GPU temp threshold", value, _GPU_TEMP_MIN, _GPU_TEMP_MAX
        )
        if n is not None:
            self.gpu_temp_threshold = n
        self.gpu_temp_threshold_error = err

    @rx.event
    def set_wandb_run_name(self, value: str) -> None:
        new, err = _apply_wandb_run_name(value)
        if new is not None:
            self.wandb_run_name = new
        self.wandb_run_name_error = err

    @rx.event
    def set_gradient_checkpointing(self, value: bool) -> None:
        self.gradient_checkpointing = bool(value)

    @rx.event
    def set_num_runs(self, value: str | int) -> None:
        n, err = _clamp_int("Num runs", value, _NUM_RUNS_MIN, _NUM_RUNS_MAX)
        if n is not None:
            self.num_runs = n
        self.num_runs_error = err

    @rx.event
    def set_samples_per_run(self, value: str | int) -> None:
        n, err = _clamp_int(
            "Samples per run", value, _SAMPLES_PER_RUN_MIN, _SAMPLES_PER_RUN_MAX
        )
        if n is not None:
            self.samples_per_run = n
        self.samples_per_run_error = err

    @rx.event
    def set_merge_mode(self, value: str) -> None:
        if value in ("slao", "simple", "ties"):
            self.merge_mode = value  # type: ignore[assignment]

    @rx.event
    def set_replay_fraction(self, value: str | float) -> None:
        f, err = _clamp_float("Replay fraction", value, 0.0, 1.0)
        if f is not None:
            self.replay_fraction = f
        self.replay_fraction_error = err

    @rx.event
    def start_multi_run(self):
        """Start a multi-run job (ui-v2 P2) through the shared job state.

        The form's own errors are refused on screen first; the JobManager
        then applies the server-side checks (sandbox, caps, one job at a
        time) and TrainState follows the job like any other.
        """
        errors = _form_errors(self) + [
            err for err in (self.num_runs_error, self.samples_per_run_error) if err
        ]
        if errors:
            return TrainState.refuse("Fix the highlighted fields first: " + "; ".join(errors))
        return TrainState.start_job(
            {
                "kind": "multi_run",
                **_training_spec_fields(self),
                "runs": int(self.num_runs),
                "samples": int(self.samples_per_run),
                "merge": str(self.merge_mode),
            }
        )


class ExportState(rx.State):
    """Export surface state: source model, format, quantization, ollama config.

    Wave 6b (FRONTEND-11): adds push_to_hub fields — backend support already
    exists in ``backpropagate.export.push_to_hub``; this state surfaces the
    inputs to the UI form.
    """

    source_model_path: str = ""
    source_model_path_error: str = ""
    format: ExportFormat = "lora"
    gguf_quant: GgufQuant = "q4_k_m"
    ollama_register: bool = False
    ollama_name: str = ""
    ollama_name_error: str = ""

    # ---- HuggingFace Hub push (FRONTEND-11 Wave 6b) ------------------------
    hub_enabled: bool = False
    hub_repo_id: str = ""
    hub_repo_id_error: str = ""
    hub_private: bool = True
    hub_branch: str = "main"
    hub_branch_error: str = ""
    # UI-A-001 (Wave A1 CRITICAL): the raw HF token is held in a BACKEND-only
    # var (``_``-prefix → Reflex never serializes it into the client WS state
    # bundle). Pre-fix ``hub_token`` was a public (base_vars) var bound
    # two-way to a type=password input, so the live write-scoped credential
    # round-tripped to the browser on every keystroke; the password mask was
    # visual only. The input is now write-only: the setter populates the
    # backend var but the field does NOT bind ``value=`` back (drop the
    # controlled-input round-trip). ``hub_token_set`` is a public BOOL mirror
    # so the form can show a "token set" affordance without echoing the
    # secret. ``hub_token_error`` stays public — it's an operator-facing
    # validation string, never the credential.
    _hub_token: str = ""  # nosec B105 — backend-only secret store, not a credential literal
    hub_token_set: bool = False
    hub_token_error: str = ""
    hub_status: str = ""  # "" / "pushing" / "done" / "error"
    hub_message: str = ""  # operator-facing status / error message

    # FRONTEND-F-004 (v1.4 Wave 6b features): surface the two CLI flags that
    # Wave 2 BRIDGE-A-004 added but the UI form was missing — ``--include-base``
    # (push merged model with base weights, not just the LoRA adapter) and
    # ``--token-file`` (read the HF token from a mode-0600 file instead of
    # the inline ``--token`` argument). The token-file path is mutually
    # exclusive with ``hub_token``; ``push_to_hub`` enforces the precedence.
    hub_include_base: bool = False
    hub_token_file_path: str = ""
    hub_token_file_path_error: str = ""

    # ---- Validated path / name setters (FRONTEND-A-002) --------------------

    @rx.event
    def set_source_model_path(self, value: str) -> None:
        """Validate and set the adapter / model path."""
        cleaned, err = _validate_ui_path(value)
        self.source_model_path = cleaned
        self.source_model_path_error = err

    @rx.event
    def set_format(self, value: str) -> None:
        if value in ("lora", "merged", "gguf"):
            self.format = value  # type: ignore[assignment]

    @rx.event
    def set_gguf_quant(self, value: str) -> None:
        if value in ("f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k"):
            self.gguf_quant = value  # type: ignore[assignment]

    @rx.event
    def set_ollama_register(self, value: bool) -> None:
        self.ollama_register = bool(value)

    @rx.event
    def set_ollama_name(self, value: str) -> None:
        """Validate the Ollama model name — alphanumeric / dash / underscore /
        colon (for tag) only; reject anything that smells like a path."""

        if not value:
            self.ollama_name = ""
            self.ollama_name_error = ""
            return
        cleaned = value.strip()
        # Ollama names: lowercase alnum + . _ - : / (the slash is for registry,
        # but we forbid backslash + .. + leading slash + null).
        if (
            ".." in cleaned
            or "\x00" in cleaned
            or "\\" in cleaned
            or cleaned.startswith("/")
            or not re.match(r"^[A-Za-z0-9._:/-]+$", cleaned)
        ):
            self.ollama_name = ""
            self.ollama_name_error = "Invalid Ollama model name"
            return
        self.ollama_name = cleaned
        self.ollama_name_error = ""

    # Live state.
    export_state: RunState = "idle"
    events: list[dict] = []

    @rx.event
    def start_export(self):
        """Start an export job (ui-v2 P2) through the shared job state.

        This is the LOCAL export-to-disk path; the HuggingFace Hub push
        (``push_to_hub`` below) is a separate handler.
        """
        errors = [
            err for err in (self.source_model_path_error, self.ollama_name_error) if err
        ]
        if errors:
            return TrainState.refuse("Fix the highlighted fields first: " + "; ".join(errors))
        return TrainState.start_job(
            {
                "kind": "export",
                "source_path": self.source_model_path,
                "export_format": str(self.format),
                "quantization": str(self.gguf_quant),
                "ollama_name": (
                    self.ollama_name
                    if self.ollama_register and self.format == "gguf"
                    else ""
                ),
            }
        )

    # ---- HuggingFace Hub push setters + handler (FRONTEND-11) --------------

    # HF repo id is ``<owner>/<repo>`` — allow [A-Za-z0-9_-./]. Strict char
    # set rejects injection probes; the 200-char cap is HF's documented
    # upper limit (see huggingface_hub.utils.validate_repo_id).
    _HF_REPO_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
    _HF_BRANCH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

    @rx.event
    def set_hub_enabled(self, value: bool) -> None:
        self.hub_enabled = bool(value)

    @rx.event
    def set_hub_repo_id(self, value: str) -> None:
        if not value or not value.strip():
            self.hub_repo_id = ""
            self.hub_repo_id_error = ""
            return
        cleaned = value.strip()
        if len(cleaned) > 200:
            self.hub_repo_id_error = "Repo id too long (max 200 chars)"
            return
        if "/" not in cleaned:
            self.hub_repo_id_error = "Repo id must be <owner>/<repo>"
            return
        if not self._HF_REPO_RE.match(cleaned):
            self.hub_repo_id_error = "Repo id: alnum / dot / dash / underscore / slash only"
            return
        self.hub_repo_id = cleaned
        self.hub_repo_id_error = ""

    @rx.event
    def set_hub_private(self, value: bool) -> None:
        self.hub_private = bool(value)

    @rx.event
    def set_hub_branch(self, value: str) -> None:
        if not value or not value.strip():
            self.hub_branch = "main"
            self.hub_branch_error = ""
            return
        cleaned = value.strip()
        if len(cleaned) > 100:
            self.hub_branch_error = "Branch name too long (max 100 chars)"
            return
        if not self._HF_BRANCH_RE.match(cleaned):
            self.hub_branch_error = "Branch: alnum / dot / dash / underscore / slash only"
            return
        self.hub_branch = cleaned
        self.hub_branch_error = ""

    @rx.event
    def set_hub_include_base(self, value: bool) -> None:
        """Toggle ``include_base`` — whether to push merged base weights too.

        FRONTEND-F-004: mirrors the ``--include-base`` CLI flag. Default
        (``False``) pushes the LoRA adapter only when the source directory
        contains adapter files; ``True`` uploads every file in the
        directory (including the base model if it's there).
        """
        self.hub_include_base = bool(value)

    @rx.event
    def set_hub_token_file_path(self, value: str) -> None:
        """Validate the HF token-file path and stage it for the push.

        FRONTEND-F-004: mirrors the ``--token-file`` CLI flag. The file is
        NOT read here — only validated for shape. ``push_to_hub`` reads the
        file at push time via the existing ``_read_hub_token_file`` helper so
        the file-mode check + POSIX-warning + empty-file error stay in one
        place.

        UI-A-001: this is the path to READ an existing user-owned credential
        file, NOT a UI WRITE target — so it is validated via
        ``_validate_token_file_path`` (expanduser + resolve + require an
        existing regular file + reject NUL bytes), NOT against the UI output
        sandbox (``get_ui_output_dir()``). The prior ``_validate_ui_path``
        forced the file under ``~/.backpropagate/ui-outputs``, which made the
        documented placeholder location ``~/.config/backpropagate/hf-token``
        (a UI-output forbidden base) impossible to enter — the field was
        unusable. Mirrors the CLI's ``_read_hub_token_file`` intent, which
        does a plain ``Path(path_str).expanduser()`` with no write-sandbox.

        Mutual exclusion with ``hub_token`` is enforced at push time
        (the inline token wins the field-clear when both are set; the
        push handler raises a structured error if both reach it).
        """
        if not value or not value.strip():
            self.hub_token_file_path = ""  # nosec B105 — path sentinel, not a password
            self.hub_token_file_path_error = ""  # nosec B105 — error-message sentinel, not a password
            return
        cleaned, err = _validate_token_file_path(value)
        self.hub_token_file_path = cleaned
        self.hub_token_file_path_error = err

    @rx.event
    def set_hub_token(self, value: str) -> None:
        """Set the HF API token (UI-A-001: write-only into a backend var).

        The input is write-only — the form does NOT bind ``value=`` back to
        this field, so the raw secret never round-trips to the client. The
        value lives ONLY in the backend var ``self._hub_token`` (never in
        base_vars / the WS bundle), is never logged, never serialized to run
        history, never echoed in error messages. ``_hub_token`` clears after
        a successful push to limit exposure. ``hub_token_set`` is a public
        bool mirror so the form can render a "token set" affordance without
        exposing the credential.
        """
        cleaned = (value or "").strip()
        if not cleaned:
            self._hub_token = ""  # nosec B105 — backend-var clear, not a credential literal
            self.hub_token_set = False
            self.hub_token_error = ""  # nosec B105 — error-message clear, not a credential literal
            return
        # HF tokens are ``hf_<40 base62 chars>``; we don't pin the exact
        # prefix because operators may use org-scoped tokens with a
        # different prefix. Sanity-check the length is in the expected
        # 30-100 char range so we catch "I pasted my username by accident".
        if len(cleaned) < 20 or len(cleaned) > 200:
            self.hub_token_error = "Token doesn't look like an HF token (20-200 chars expected)"  # nosec B105 — operator-facing validation message, not a credential
            self.hub_token_set = False
            return
        self._hub_token = cleaned  # nosec B105 — backend-only secret store
        self.hub_token_set = True
        self.hub_token_error = ""  # nosec B105 — error-message clear, not a credential literal

    @rx.event
    def push_to_hub(self) -> None:
        """Push the trained adapter / merged model to a HuggingFace Hub repo.

        Delegates to ``backpropagate.export.push_to_hub`` (the established
        backend API). Failures surface via ``self.hub_message``; success
        clears the token so it doesn't sit in the state for the lifetime of
        the WS session.

        Pre-flight validation:
        - source_model_path must be set
        - hub_repo_id + hub_token must be set
        - all field-level errors must be clear

        The handler runs synchronously; the operator sees ``hub_status =
        "pushing"`` for the duration. v1.4 should move this to a background
        task with an SSE progress stream.
        """
        if self.source_model_path == "":
            self.hub_status = "error"
            self.hub_message = "Set a source adapter / model path before pushing."
            return
        if not self.hub_repo_id or self.hub_repo_id_error:
            self.hub_status = "error"
            self.hub_message = "Set a valid HuggingFace repo id (<owner>/<repo>)."
            return
        # FRONTEND-F-004: mutual-exclusion + at-least-one check on the two
        # token surfaces. Mirrors the CLI's `--token` vs `--token-file`
        # contract in cmd_push (cli.py ~3411).
        inline_token_set = bool(self._hub_token) and not self.hub_token_error
        token_file_set = (
            bool(self.hub_token_file_path) and not self.hub_token_file_path_error
        )
        if inline_token_set and token_file_set:
            self.hub_status = "error"
            self.hub_message = (
                "Token and Token-file are mutually exclusive — clear one "
                "before pushing. The token-file path is the safer floor "
                "(mode 0600, not visible to spawned children)."
            )
            return
        if not inline_token_set and not token_file_set:
            self.hub_status = "error"
            self.hub_message = (
                "Set a valid HuggingFace API token (inline) or a token-file "
                "path before pushing."
            )
            return
        if self.hub_branch_error:
            self.hub_status = "error"
            self.hub_message = "Fix the branch field error before pushing."
            return
        if self.hub_token_file_path_error:
            self.hub_status = "error"
            self.hub_message = "Fix the Token-file path error before pushing."
            return

        self.hub_status = "pushing"
        self.hub_message = (
            f"Pushing {self.source_model_path} to "
            f"{self.hub_repo_id}@{self.hub_branch or 'main'}…"
        )
        try:
            from .export import push_to_hub as _push

            # FRONTEND-F-004: resolve the token at push time. The
            # token-file path is read via the shared CLI helper so the
            # mode-0600 warning + empty-file error live in one spot.
            resolved_token: str
            if token_file_set:
                from .cli import _read_hub_token_file

                resolved_token = _read_hub_token_file(
                    self.hub_token_file_path,
                    flag_name="--token-file (UI)",
                )
            else:
                resolved_token = self._hub_token

            _push(
                local_path=self.source_model_path,
                repo_id=self.hub_repo_id,
                token=resolved_token,
                private=bool(self.hub_private),
                revision=(self.hub_branch or "main"),
                include_base=bool(self.hub_include_base),
            )
            self.hub_status = "done"
            self.hub_message = (
                f"Pushed to https://huggingface.co/{self.hub_repo_id} "
                f"on branch {self.hub_branch or 'main'}."
            )
            # Clear token after successful push (UI-A-001: backend var).
            self._hub_token = ""  # nosec B105 — token wipe after push, not a credential literal
            self.hub_token_set = False
            # FRONTEND-F-004: the token-file PATH itself is not a credential
            # — it's a reference to a file the operator manages outside
            # the UI session. Leaving it in state is intentional so a
            # second push (e.g. after a transient HF outage) doesn't
            # require re-typing the path.
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # Sanitize the error so HF token / operator paths don't leak.
            try:
                from .ui_security import sanitize_error_for_user

                message, suggestion = sanitize_error_for_user(
                    exc, operation="pushing to HuggingFace Hub"
                )
                self.hub_status = "error"
                self.hub_message = (
                    message + (f" Try: {suggestion}" if suggestion else "")
                )
            except Exception:  # noqa: BLE001
                # Last-resort fallback — preserve the operator-facing
                # message but trim to 200 chars so we don't spill a giant
                # traceback into the WS bundle. V2-a (sibling): redact absolute
                # paths from the raw exception repr — ``hub_message`` is a
                # public (client-serialized) var, so an HF exception embedding
                # the operator's home dir would otherwise ship to the browser.
                self.hub_status = "error"
                self.hub_message = _redact_action(
                    f"Push failed: {type(exc).__name__}: {str(exc)[:200]}"
                )

    @rx.event
    def clear_hub_status(self) -> None:
        """Dismiss the HF push status banner."""
        self.hub_status = ""
        self.hub_message = ""


class DatasetState(rx.State):
    """Dataset surface state: upload, format detect, preview, dedup config."""

    # UI-A-002 (Wave A2): the full upload path embeds the operator's home dir
    # + username and was previously a PUBLIC (client-serialized) Reflex var —
    # so the home prefix shipped in the WS bundle on every state delta even
    # though FRONTEND-B-013 only RENDERED the basename. The full path is now
    # held in a backend-only ('_'-prefixed) var that Reflex never serializes;
    # the client sees only the basename via the ``uploaded_basename`` computed
    # var. The full path stays available server-side for the Trainer hookup.
    _uploaded_path: str = ""
    upload_error: str = ""
    upload_count: int = 0  # per-session cap
    detected_format: str = ""
    preview_records: list[dict] = []

    # FRONTEND-B-013 / UI-A-002: backend-computed basename so the UI never has
    # to split the full path on the client AND the full path (with home
    # prefix) never enters the serialized state bundle. Only the basename is
    # rendered.
    @rx.var
    def uploaded_basename(self) -> str:
        if not self._uploaded_path:
            return ""
        return Path(self._uploaded_path).name

    @rx.var
    def has_upload(self) -> bool:
        """Client-safe truthiness for the 'Uploaded: …' chrome.

        The template can't test the backend-only ``_uploaded_path`` (it isn't
        serialized), so expose a boolean computed var for the ``rx.cond``.
        """
        return bool(self._uploaded_path)

    # Format hint — operator can override the auto-detect when it guesses wrong.
    format_hint: DatasetFormatHint = "auto"

    # Dedup + filter knobs.
    dedup_enabled: bool = True
    drop_empty: bool = True
    apply_curriculum: bool = False
    min_tokens: int = 0
    min_tokens_error: str = ""
    max_tokens: int = 2048
    max_tokens_error: str = ""

    record_count: int = 0
    dedup_hits: int = 0
    # FRONTEND-F-005 (v1.4 Wave 6b features): plumb the Stats-grid value
    # that dataset.py was hardcoding as '—'. The figure comes from
    # ``backpropagate.datasets.get_dataset_stats`` during the upload
    # handler — same computation the CLI emits, just thread it to the UI
    # state so the grid doesn't drift from the backend truth.
    avg_tokens: int = 0

    # Per-session upload cap. Reflex state is per-WebSocket-connection so this
    # is effectively per-tab; an unauthenticated abuser can still open many
    # tabs (cf. FRONTEND-A-001), but the cap caps each session's foot-gun.
    _MAX_UPLOADS_PER_SESSION: int = 5

    @rx.event
    async def handle_upload(self, files: list) -> None:  # type: ignore[type-arg]
        """Validate and persist uploaded dataset files (FRONTEND-A-003).

        The handler is wired to ``rx.upload``'s ``on_drop``. Each file goes
        through ``FileValidator`` (extension allowlist + size cap + magic-byte
        sniff when enabled) and ``sanitize_filename`` before being written
        inside ``get_ui_output_dir() / 'uploads'``. Failures populate
        ``upload_error`` rather than raising — the UI binds to it via
        ``rx.cond``.
        """
        from .ui_security import (
            ALLOWED_DATASET_EXTENSIONS,
            DEFAULT_SECURITY_CONFIG,
            FileValidator,
            get_ui_output_dir,
            sanitize_filename,
        )

        # FRONTEND-B-006: defense-in-depth against a malicious WS client that
        # sends multiple files in one ``on_drop`` payload despite ``multiple=
        # False`` on the rx.upload widget. The entry check below is fast-fail;
        # the per-iteration check inside the loop is the real cap enforcement.
        if not isinstance(files, list) or len(files) == 0:
            self.upload_error = "No files received"
            return
        if len(files) > 1:
            # Server contract is multiple=False; a payload with >1 file is a
            # WebSocket-direct bypass attempt. Reject the whole drop rather
            # than partially processing it.
            self.upload_error = (
                "Only one file per upload (multiple-file drops are rejected)"
            )
            return
        if self.upload_count >= self._MAX_UPLOADS_PER_SESSION:
            self.upload_error = (
                f"Per-session upload cap reached "
                f"({self._MAX_UPLOADS_PER_SESSION} files). "
                "Restart the page to upload more."
            )
            return

        try:
            base = get_ui_output_dir()
        except Exception as exc:  # noqa: BLE001
            # FRONTEND-B-007 (Stage C humanization): route the raw OSError /
            # BackpropagateError through sanitize_error_for_user so the UI
            # banner cannot leak operator paths (FB-011 invariant).
            from .ui_security import sanitize_error_for_user

            message, suggestion = sanitize_error_for_user(
                exc, operation="resolving the UI output directory"
            )
            self.upload_error = (
                message + (f" Try: {suggestion}" if suggestion else "")
            )
            return

        upload_dir = base / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        validator = FileValidator(
            allowed_extensions=ALLOWED_DATASET_EXTENSIONS,
            max_size_mb=DEFAULT_SECURITY_CONFIG.max_upload_size_mb,
        )
        max_bytes = DEFAULT_SECURITY_CONFIG.max_upload_size_mb * 1024 * 1024

        for f in files:
            # FRONTEND-B-006: per-iteration cap check — if the multiple-file
            # guard above is ever weakened (e.g. for a future drag-multiple
            # feature) this is the floor that still holds.
            if self.upload_count >= self._MAX_UPLOADS_PER_SESSION:
                self.upload_error = (
                    f"Per-session upload cap reached "
                    f"({self._MAX_UPLOADS_PER_SESSION} files)."
                )
                return
            filename = getattr(f, "filename", None) or "unnamed"
            # FRONTEND-B-004 (Stage C humanization): stream-read in fixed-size
            # chunks with a running size counter so the cap is enforced
            # incrementally rather than after a full in-memory buffer. Peak
            # memory per upload is bounded at max_bytes + _CHUNK regardless of
            # what the client sends; a 10 GB rogue payload claiming to be a
            # .jsonl is rejected after the first chunk over the cap rather
            # than buffering the whole 10 GB.
            #
            # The fallback path covers reader objects that lack a chunked
            # async read (the rx.upload framing exposes ``await f.read()`` as
            # an all-bytes call; if the underlying reader is something else,
            # we still bail safely).
            _CHUNK = 1 << 20  # 1 MB per chunk
            chunks: list[bytes] = []
            size = 0
            try:
                # Probe for chunked-read support. ``read(n)`` returning bytes
                # is the asyncio.StreamReader contract; rx.upload's wrapper
                # supports it as of Reflex 0.9.x.
                while True:
                    chunk = await f.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        # Drop already-buffered chunks; we never persist a
                        # rejected upload.
                        chunks.clear()
                        self.upload_error = (
                            f"Rejected {filename}: exceeds "
                            f"{DEFAULT_SECURITY_CONFIG.max_upload_size_mb} MB "
                            "cap (aborted mid-stream). Trim the file or "
                            "increase BACKPROPAGATE_UI__MAX_UPLOAD_SIZE_MB."
                        )
                        return
                    chunks.append(chunk)
            except TypeError:
                # Reader doesn't honor a chunk-size argument (e.g. a
                # one-shot bytes object surfaced as a fake reader); fall
                # back to a single read and the post-buffer size check.
                try:
                    data = await f.read()
                except Exception as exc:  # noqa: BLE001
                    from .ui_security import sanitize_error_for_user

                    message, suggestion = sanitize_error_for_user(
                        exc, operation=f"reading upload '{filename}'"
                    )
                    self.upload_error = (
                        message
                        + (f" Try: {suggestion}" if suggestion else "")
                    )
                    return
                if len(data) > max_bytes:
                    self.upload_error = (
                        f"Rejected {filename}: exceeds "
                        f"{DEFAULT_SECURITY_CONFIG.max_upload_size_mb} MB cap. "
                        "Trim the file or increase "
                        "BACKPROPAGATE_UI__MAX_UPLOAD_SIZE_MB."
                    )
                    return
                chunks = [data]
            except Exception as exc:  # noqa: BLE001
                # FRONTEND-B-007 (Stage C humanization): never echo raw OSError
                # messages into the UI banner — operator paths leak via the
                # exception repr.
                from .ui_security import sanitize_error_for_user

                message, suggestion = sanitize_error_for_user(
                    exc, operation=f"reading upload '{filename}'"
                )
                self.upload_error = (
                    message + (f" Try: {suggestion}" if suggestion else "")
                )
                return

            data = b"".join(chunks)

            # Stage to a temp file so FileValidator can run its file-on-disk
            # checks (extension + size + magic-bytes when enabled).
            safe_name = sanitize_filename(filename)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(safe_name).suffix
            ) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            try:
                # Adapt to FileValidator's "object with .name" contract.
                class _FObj:
                    name = str(tmp_path)

                is_valid, msg, _ = validator.validate(_FObj(), purpose="upload")
                if not is_valid:
                    self.upload_error = f"Rejected {filename}: {msg}"
                    return
            finally:
                # We persist data ourselves under the allowed base; drop temp.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

            target = upload_dir / safe_name
            target.write_bytes(data)
            # UI-A-002: store the full path in the backend-only var; the
            # client receives only the basename via ``uploaded_basename``.
            self._uploaded_path = str(target)
            self.upload_count += 1

            # FRONTEND-F-005 (v1.4 Wave 6b features): compute dataset stats
            # via the canonical backend API + thread them into state so the
            # Stats grid (record_count / avg_tokens / dedup_hits) and the
            # Format badge bind to live values rather than hardcoded '—'
            # placeholders. The computation matches what
            # `backprop validate-dataset` emits, so UI + CLI stay in
            # lockstep. Failure here is non-fatal — the file is already
            # persisted; we just leave the Stats grid empty and surface a
            # one-line note inside the same upload_error channel.
            try:
                import json as _json

                from .datasets import (
                    DatasetFormat,
                    _detect_format_from_file,
                    get_dataset_stats,
                )

                samples: list[dict | str] = []
                suffix = target.suffix.lower()
                if suffix == ".jsonl":
                    with target.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    samples.append(_json.loads(line))
                                except _json.JSONDecodeError:
                                    # Skip malformed lines in stats —
                                    # the strict validate-dataset
                                    # surface catches these.
                                    continue
                elif suffix == ".json":
                    with target.open(encoding="utf-8") as fh:
                        parsed = _json.load(fh)
                        if isinstance(parsed, list):
                            samples = parsed
                        else:
                            samples = [parsed]
                # Other extensions: leave samples empty → stats render 0s.

                detected = _detect_format_from_file(target)
                stats = get_dataset_stats(samples, format_type=detected)
                self.record_count = int(stats.total_samples)
                # avg_tokens_per_sample is float; round to int for the
                # 4-size big-number cell. Sub-token resolution isn't
                # operator-useful at this glance-level.
                self.avg_tokens = int(round(stats.avg_tokens_per_sample))
                if detected != DatasetFormat.UNKNOWN:
                    # Map enum to a short human badge — the Format group
                    # already binds ``detected_format`` and is None-safe.
                    self.detected_format = str(detected.value).capitalize()
            except Exception:  # noqa: BLE001  # nosec B110 — stats are advisory, never block upload
                # Leave the grid empty rather than raising. The upload
                # itself succeeded; a stats failure shouldn't surface as
                # an upload error.
                pass

        self.upload_error = ""

    @rx.event
    def set_format_hint(self, value: str) -> None:
        if value in ("auto", "sharegpt", "alpaca", "openai", "jsonl"):
            self.format_hint = value  # type: ignore[assignment]

    @rx.event
    def set_dedup_enabled(self, value: bool) -> None:
        self.dedup_enabled = bool(value)

    @rx.event
    def set_drop_empty(self, value: bool) -> None:
        self.drop_empty = bool(value)

    @rx.event
    def set_apply_curriculum(self, value: bool) -> None:
        self.apply_curriculum = bool(value)

    @rx.event
    def set_min_tokens(self, value: str | int) -> None:
        n, err = _clamp_int("Min tokens", value, _TOKENS_MIN, _TOKENS_MAX)
        if n is not None:
            self.min_tokens = n
            # Re-clamp max if min would exceed it.
            if self.max_tokens < n:
                self.max_tokens = n
        self.min_tokens_error = err

    @rx.event
    def set_max_tokens(self, value: str | int) -> None:
        n, err = _clamp_int("Max tokens", value, _TOKENS_MIN, _TOKENS_MAX)
        if n is not None:
            self.max_tokens = n
        self.max_tokens_error = err

    @rx.event
    def detect_format_stub(self) -> None:
        """Stub handler — placeholder for the upload->detect flow."""
        if self._uploaded_path:
            self.detected_format = "alpaca"


# ---------------------------------------------------------------------------
# RunsState — backs the /runs page (FRONTEND-F-RUN-HISTORY-PAGE, Wave 6)
# ---------------------------------------------------------------------------
#
# Loads the recent training-run history via the CLI's RunHistoryManager so the
# UI shows the same data ``backprop list-runs`` shows. The implementation
# imports RunHistoryManager directly rather than shelling out to the CLI —
# subprocess-shelling from a Reflex state handler would block the WS event
# loop and the CLI prints decorated text, not the clean dicts the UI needs.
#
# Drill-down to a per-run page is INTENTIONALLY OUT OF SCOPE for v1.2.0 (the
# user narrowed the brief). Each table row is a read-only summary; v1.3 adds
# a /runs/<id> route that mirrors ``backprop show-run``.


class RunsState(rx.State):
    """Run-history surface state — populates the /runs page table."""

    runs: list[dict] = []
    loading: bool = False
    error: str = ""
    # HUX-02 (Stage C humanization): the operator-actionable remedy from
    # ``sanitize_error_for_user`` is held SEPARATELY from ``error`` (not folded
    # into it as a run-on "… Try: {suggestion}") so the /runs error callout can
    # render it on its own dimmed line via the ``BpErrorCallout`` ``hint=``
    # slot. Empty when there is no error or the error carries no suggestion.
    error_suggestion: str = ""
    status_filter: str = ""  # "" / running / completed / failed
    output_dir_override: str = ""
    last_loaded_at: str = ""
    last_loaded_label: str = ""

    # Hard cap on rows rendered at once. The CLI defaults to 50; the table can
    # comfortably render this many without pagination. v1.3 will add a
    # "Load more" affordance + per-status filter pills.
    _DEFAULT_LIMIT: int = 50

    @rx.event
    def load_runs(self) -> None:
        """Populate ``self.runs`` from the on-disk run history.

        The default output directory is ``~/.backpropagate/ui-outputs`` (the
        same directory the UI writes adapters/exports into). Operators who
        train from the CLI to a different ``--output`` directory can set the
        ``output_dir_override`` field on this state before calling
        ``load_runs``; the UI's settings surface will wire that in v1.3.
        """
        from datetime import datetime, timezone
        from pathlib import Path as _Path

        self.loading = True
        self.error = ""
        self.error_suggestion = ""
        try:
            # Resolve the history directory. Use the override if set, otherwise
            # fall back to the UI's own output dir (the default training sink).
            if self.output_dir_override.strip():
                # UI-A-005 (Wave A2): ``output_dir_override`` is a PUBLIC
                # (client-serialized) Reflex var. ``set_output_dir_override``
                # validates via ``_validate_ui_path`` — but a malicious WS
                # client can write the public var directly, bypassing the
                # setter. Re-validate the forbidden-base guard HERE, at the
                # read sink, so a value pointing at a system / credential dir
                # (``/etc``, ``~/.ssh``, ``C:\\Windows`` …) is refused even if
                # it never passed through the setter. This mirrors the env-var
                # guard ``get_ui_output_dir()`` applies to its override.
                history_dir = _Path(self.output_dir_override).expanduser()
                try:
                    from .ui_security import _is_forbidden_output_base

                    if _is_forbidden_output_base(history_dir):
                        self.runs = []
                        self.error = _redact_action(
                            "Refusing to read run history from a system or "
                            f"credential directory: {history_dir.resolve()}. "
                            "Point the override at a non-system directory."
                        )
                        return
                except Exception:  # noqa: BLE001 — guard import/resolve must
                    # fail closed: if we can't validate, refuse the override
                    # and fall back to the sandbox default rather than reading
                    # an unvalidated operator-supplied path.
                    try:
                        from .ui_security import get_ui_output_dir

                        history_dir = get_ui_output_dir()
                    except Exception:
                        history_dir = _Path.home() / ".backpropagate" / "ui-outputs"
            else:
                try:
                    from .ui_security import get_ui_output_dir

                    history_dir = get_ui_output_dir()
                except Exception:
                    # Final fallback: the documented default.
                    history_dir = _Path.home() / ".backpropagate" / "ui-outputs"

            if not history_dir.exists():
                self.runs = []
                # UI-A-002 (sibling): ``error`` is a public RunsState var;
                # history_dir embeds the home dir + username. Redact.
                self.error = _redact_action(
                    f"No run history at {history_dir}. Train a model from the "
                    "UI or CLI; runs will appear here automatically."
                )
                return

            try:
                from .checkpoints import RunHistoryManager
            except ImportError as exc:
                self.error = _redact_action(f"checkpoints module unavailable: {exc}")
                self.runs = []
                return

            manager = RunHistoryManager(str(history_dir))
            status = self.status_filter.strip() or None
            try:
                rows = manager.list_runs(status=status, limit=self._DEFAULT_LIMIT)
                # ui-v2 P1 fix round: UI-spawned runs keep their history in
                # jobs/<run>/output/run_history.json — merge them in so the
                # Runs page lists UI-driven training, not just CLI runs.
                rows = _merge_job_history_rows(
                    rows, history_dir, status=status, limit=self._DEFAULT_LIMIT
                )
            except ValueError as exc:
                # ValueError is operator-actionable (bad filter value); the
                # exception message itself is shaped for display.
                self.error = f"Invalid filter: {exc}"
                self.runs = []
                return
            except Exception as exc:  # noqa: BLE001 — surface as operator string
                # FRONTEND-B-007 (Stage C humanization): route through
                # sanitize_error_for_user so raw OSError / JSONDecodeError
                # messages (which embed filesystem paths) don't leak into the
                # UI banner. The full traceback is still logged server-side
                # via the caller's logger.exception (FB-011 invariant).
                from .ui_security import sanitize_error_for_user

                message, suggestion = sanitize_error_for_user(
                    exc, operation="loading run history"
                )
                # HUX-02: keep the (message, suggestion) split — the remedy
                # rides the separate ``error_suggestion`` var (rendered as a
                # scannable dimmed ``hint=`` line in the callout) rather than
                # being concatenated into ``error`` as a run-on sentence.
                self.error = message
                self.error_suggestion = suggestion or ""
                self.runs = []
                return

            # Normalize to a small, JSON-serializable shape for Reflex's WS
            # bundle. The CLI emits dicts already; we just trim fields the
            # table doesn't use so the bundle stays small. We also pre-format
            # the run_id to its short 8-char form to avoid an f-string in
            # the template (Reflex template f-strings get awkward fast).
            trimmed: list[dict] = []
            for raw in rows:
                run_id = str(raw.get("run_id") or "")
                short_id = run_id[:8] if run_id else "-"
                started = _fmt_started(raw.get("started_at"))
                duration = raw.get("duration_seconds")
                if duration is None:
                    duration_str = "-"
                else:
                    try:
                        duration_str = f"{float(duration):.0f}s"
                    except (TypeError, ValueError):
                        duration_str = "-"
                final_loss = raw.get("final_loss")
                if final_loss is None:
                    final_loss_str = "-"
                else:
                    try:
                        final_loss_str = f"{float(final_loss):.4f}"
                    except (TypeError, ValueError):
                        final_loss_str = "-"
                trimmed.append({
                    "run_id": run_id,
                    "run_id_short": short_id,
                    "started_at": started,
                    # RunHistoryManager stores ``model_name`` / ``dataset_info``
                    # (the CLI's list-runs maps them to short keys itself); fall
                    # back to the short keys for older / hand-built entries. The
                    # dataset is usually an absolute path, so redact it before it
                    # enters this client-serialized var (as RunDetailState does).
                    "model": str(raw.get("model_name") or raw.get("model") or "-"),
                    "dataset": _dataset_label(
                        raw.get("dataset_info") or raw.get("dataset")
                    ),
                    "status": str(raw.get("status") or "-"),
                    "duration": duration_str,
                    "final_loss": final_loss_str,
                })
            self.runs = trimmed
            self.last_loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.last_loaded_label = _fmt_local_time()
        finally:
            self.loading = False

    # Canonical status set. Mirrors the values RunHistoryManager.list_runs
    # accepts (``VALID_STATUSES``: running / completed / failed - it raises
    # ValueError for anything else, so offering e.g. "interrupted" here made
    # that dropdown choice always error); the dropdown in pages/runs.py renders
    # the same set. Update both surfaces together when adding a new status.
    _STATUS_FILTER_VALUES: tuple[str, ...] = (
        "",
        "running",
        "completed",
        "failed",
    )

    @rx.event
    def set_status_filter(self, value: str) -> None:
        """Update the status filter and reload.

        Unknown values are silently discarded (the previous filter persists)
        but logged at WARNING so the silent-drop is observable in operator
        logs - FRONTEND-B-014 (Stage C humanization). A future status added
        in RunHistoryManager but missing from this list would otherwise
        present as 'filter does nothing' with no breadcrumb.

        ``load_runs`` is only invoked when the value is accepted - the
        previous code unconditionally reloaded even on rejection, which
        produced a confusing double-trigger of the table.
        """
        if value in self._STATUS_FILTER_VALUES:
            self.status_filter = value
            self.load_runs()
            return
        import logging

        logging.getLogger(__name__).warning(
            "RunsState: status filter received unknown value %r "
            "(keeping previous %r). Allowed values: %s",
            value,
            self.status_filter,
            ", ".join(repr(v) for v in self._STATUS_FILTER_VALUES),
        )

    @rx.event
    def set_output_dir_override(self, value: str) -> None:
        """Operator-supplied output directory (validated lightly)."""
        cleaned, err = _validate_ui_path(value)
        if err:
            self.error = err
            return
        self.output_dir_override = cleaned

    @rx.event
    def clear_error(self) -> None:
        """Dismiss the error banner."""
        self.error = ""
        self.error_suggestion = ""


# ---------------------------------------------------------------------------
# AuthBadgeState - backs the footer auth-badge UI
# (FRONTEND-F-FOOTER-AUTH-BADGE, Stage C humanization).
# ---------------------------------------------------------------------------
#
# Reads ``ui_security.get_auth_badge_context()`` at first access and exposes
# the 6 fields the footer chip needs. The state is server-side only - the
# fields are populated from env vars that the CLI exports BEFORE spawning the
# Reflex subprocess, so the values are stable for the lifetime of the UI
# process. No event handlers mutate the fields.
#
# Critically: this class READS the auth posture; it does NOT participate in
# auth enforcement. The GHSA-f65r-h4g3-3h9h contracts (pre-accept WS cookie
# validation, 4-mode resolution, constant-time compares, Host/Origin
# allowlists) remain the exclusive responsibility of
# ``ui_app/auth.py::basic_auth_transformer``.


class AuthBadgeState(rx.State):
    """State backing the footer auth-mode badge.

    All 6 fields are populated at class-init from the env-var surface the
    middleware also consumes (``ui_security.get_auth_badge_context``). They
    are read-only from the operator's perspective: the badge reflects the
    posture the CLI established at launch.
    """

    mode_key: str = ""
    mode_color: str = "green"
    mode_text: str = ""
    hover_text: str = ""
    bind_host: str = ""
    bind_port: str = ""
    reachable_from: str = ""
    # FRONTEND-A-002 (v1.4 Wave 2): wire ``ctx.auth_user`` end-to-end.
    # Pre-fix the field was populated on the context dataclass but discarded
    # in ``refresh()``; the auth-mode label said "Basic" but the operator
    # could not see WHICH credential pair was active from the badge alone
    # (had to hover for the tooltip text). Surfacing the username as a
    # separate chip mirrors the bind-host chip pattern from Wave 5.5 and
    # closes the producer-without-consumer dead-state.
    auth_user: str = ""

    @rx.event
    def refresh(self) -> None:
        """Recompute the badge state from the current env.

        Mounted on the chrome's footer ``on_mount`` so a refreshed env var
        (rare - the CLI exports them once at launch) is reflected without a
        process restart. The cost is one ``os.environ`` snapshot per page
        load, which is dwarfed by every other request the Reflex backend
        handles.

        FRONTEND-B-006 (v1.4 Wave 4 Stage C humanization): the footer renders
        on EVERY route, so ``on_mount=AuthBadgeState.refresh`` previously
        fired on every page navigation. The env vars are stable for the
        process lifetime (the CLI exports them once at launch), so re-
        snapshotting on every nav is wasted server work + a small WS round-
        trip that flickers the chip on slow connections. We now early-exit
        when ``mode_key`` is already populated, making the badge refresh
        once-per-WS-session in steady state. The CLI is free to call
        ``refresh()`` explicitly if it rotates env vars mid-process; the
        early-exit only suppresses the per-nav repeat.
        """
        if self.mode_key:
            # Badge state was populated on the first footer mount of this
            # WS session; the env-var surface is stable so the second mount
            # would produce identical state. Skip the snapshot.
            return

        from .ui_security import get_auth_badge_context

        ctx = get_auth_badge_context()
        self.mode_key = ctx.mode_key
        self.mode_color = ctx.mode_color
        self.mode_text = ctx.mode_text
        self.hover_text = ctx.hover_text
        self.bind_host = ctx.bind_host
        self.bind_port = ctx.bind_port
        self.reachable_from = ctx.reachable_from
        # FRONTEND-A-002 (v1.4 Wave 2): mirror ``ctx.auth_user`` into state so
        # the footer badge can render it as a visible "@username" suffix on
        # the three Basic-auth modes (basic_local / basic_shared /
        # basic_network). The non-Basic modes (no_auth_local / token_local /
        # insecure) populate this with the empty string and the badge
        # component skips the chip via ``rx.cond``.
        self.auth_user = ctx.auth_user


# ---------------------------------------------------------------------------
# RunDetailState — backs /runs/[rid] (Wave 6b drill-down)
# ---------------------------------------------------------------------------
#
# Wave 6 shipped the read-only run list; Wave 6b adds the drill-down per
# V1_3_BRIEF P1. The state mirrors what ``backprop show-run`` would emit
# (metadata header + hyperparameter dump + training metrics + checkpoint
# list + log tail) using the existing ``RunHistoryManager.get_run`` API
# (which supports partial-prefix matching for operator convenience).
#
# Of the four action buttons, only Diff shells out — to ``backprop
# diff-runs`` (the bridge owns that subcommand; the handler dispatches and
# renders the output). Replay, Delete, and Export run IN-PROCESS via
# ``RunHistoryManager`` (UI-A-004): their prior shell-outs targeted
# phantom CLI surfaces (no ``delete-run`` subcommand, no ``--run-id`` on
# ``export-runs``, no ``--dry-run`` on ``replay``) and always failed, so
# they were rewired to do the work directly and re-load on completion.


class RunDetailState(rx.State):
    """Per-run drill-down state.

    The active run id is read from the dynamic route parameter inside
    ``load_run`` (Reflex 0.9's ``self.router.page.params.get("rid")``
    pattern — the route is ``/runs/[rid]``). We name the route arg ``rid``
    (not ``run_id``) because Reflex refuses to bind a dynamic route arg that
    shadows an existing state var (DynamicRouteArgShadowsStateVarError); the
    state field is named ``current_run_id`` and the route arg writes to it on
    mount.

    The remaining fields are filled by ``load_run`` on mount.
    """

    current_run_id: str = ""
    loading: bool = False
    error: str = ""
    not_found: bool = False
    # FRONTEND-B-001 (v1.4 Wave 3.5): a successful ``delete_run`` shell-out
    # leaves the page showing data for a run that no longer exists on disk.
    # Pre-fix the handler set ``not_found = True`` to suppress the stale
    # body — but that latched the operator into the "Run not found" chrome
    # (designed for unknown-id navigations), which misframed a successful
    # operation as failure and stranded the success message in
    # ``action_result``. ``was_deleted`` is a separate, deletion-specific
    # surface so the template can render a "Run deleted." confirmation
    # with a "Back to runs list" affordance instead of the not-found
    # chrome. The template branches on ``was_deleted`` BEFORE
    # ``not_found`` so a successful delete always wins.
    was_deleted: bool = False

    # Header fields
    status: str = "-"
    model: str = "-"
    dataset: str = "-"
    started_at: str = "-"
    completed_at: str = "-"
    duration: str = "-"
    final_loss: str = "-"
    # UI-A-002 (Wave A2): the checkpoint path embeds the operator's home dir +
    # username. Held in a backend-only var; the client sees only the redacted
    # form via the ``checkpoint_path_display`` computed var (home prefix
    # replaced with ``<redacted-path>`` so the operator still sees the
    # run-relative tail but not their username).
    _checkpoint_path: str = "-"
    # ui-v2 P2: the folder "Export the model" hands to the Export page — a
    # UI job's <job>/output (where its adapter is saved), else the recorded
    # checkpoint path. Backend-only; ``can_export_model`` is the public flag.
    _export_source: str = ""
    can_export_model: bool = False

    # Hyperparameter table — list of {key, value} dicts so Reflex's foreach
    # can render them as table rows without on-template f-strings.
    hyperparameters: list[dict] = []

    # Metrics — read from training_metrics.jsonl when present. v1.4 ships
    # with loss_history only. The multi-line metrics view (V1_4_BRIEF item
    # 10: lr + grad_norm + val_loss as additional series) was DEFERRED to
    # v1.5 per advisor lock 2026-05-25 (Wave 5 feature audit, decision 5)
    # because the upstream data pipeline is dead: trainer.py's log-history
    # extraction reads only ``log.get('loss')`` from HF Trainer.state.log_
    # history, dropping the ``grad_norm`` / ``learning_rate`` keys HF
    # populates per step; checkpoints.py's manifest schema has no parallel
    # ``grad_norm_history`` / ``lr_history`` / ``val_loss_history`` fields.
    # v1.5 will land the full cohesive slice in one wave — trainer
    # extraction + schema bump + RunDetailState fields + BpLossChart
    # multi-line + dual-axis y-scale + per-series toggle — rather than
    # ship the data plumbing as a banner-documenting-no-op intermediate
    # in v1.4 (see [[no-banner-documenting-no-op]]).
    loss_history: list[float] = []

    # Checkpoint list — {name, size_mb, timestamp} per entry.
    checkpoints: list[dict] = []

    # Log tail — last 200 lines of training.log when present (trimmed for
    # WS bundle size). UI-A-002: each line is run through ``_redact_paths``
    # before assignment in ``load_run`` so absolute paths a training log may
    # contain (the checkpoint dir, the HF cache, tempdirs) don't ship the
    # operator's home dir + username to the client.
    log_lines: list[str] = []

    # UI-A-002 (Wave A2): client-facing, redacted form of the checkpoint
    # path. The full path lives in the backend-only ``_checkpoint_path``;
    # this computed var replaces the home prefix with ``<redacted-path>`` so
    # the operator still sees the run-relative tail without leaking their
    # username into the WS bundle / screenshots.
    @rx.var
    def checkpoint_path_display(self) -> str:
        if not self._checkpoint_path or self._checkpoint_path == "-":
            return "-"
        return _redact_action(self._checkpoint_path)

    # Action panel — last action result (for the operator-facing toast).
    # Diff shells out to ``backprop diff-runs``; Replay / Delete / Export run
    # in-process via RunHistoryManager.
    action_result: str = ""
    action_error: str = ""
    # HUX-02 (Stage C humanization): operator-actionable remedy for the most
    # recent action failure, held separately from ``action_error`` so the
    # run-detail callout can render it on its own dimmed ``hint=`` line instead
    # of folding it into the message as a run-on. RunDetailState's action
    # errors are currently all path-redacted informational strings (no
    # ``sanitize_error_for_user`` suggestion split), so this stays empty today;
    # the slot is wired backward-compatibly for a future suggestion-bearing
    # action error and keeps the three error callouts (/runs, /models,
    # /run-detail) on one scannable hierarchy.
    action_error_suggestion: str = ""
    # FRONTEND-B-014-EXTENDED (v1.4 Wave 4 Stage C humanization): action-in-
    # flight state for the diff / replay / delete / export actions. The
    # handlers run synchronously and can block (the diff-runs subprocess up to
    # 30s; the in-process replay / delete / export are bounded by disk I/O).
    # Pre-fix the operator clicked the button and saw NO feedback until the
    # handler returned — a long, silent gap that read as a frozen UI. The
    # action panel now branches on this Var to render an inline spinner +
    # "Running …" copy so the operator knows the action is in flight. Set to a
    # short human label (e.g. ``"diff-runs"``) at the start of each handler and
    # cleared at the end via try/finally.
    action_in_flight: str = ""

    # FRONTEND-A-001 (v1.4 Wave 2): comparison run id for the Diff button.
    # Pre-fix ``diff_against`` was a fully-implemented handler with no UI
    # control invoking it (the brief promised 4 action buttons; only 3
    # shipped). The Diff form on ``run_detail.py:_action_panel`` writes into
    # this field via ``set_diff_other_run_id``; the Compare button calls
    # ``diff_against`` with the current value. Form lives next to the
    # primary action row so the operator never leaves the page to compare.
    diff_other_run_id: str = ""

    @rx.event
    def set_diff_other_run_id(self, value: str) -> None:
        """Update the comparison-run-id text input (FRONTEND-A-001).

        UI-A-003 (Wave A1 HIGH): validate at the trust boundary. The value
        flows into ``subprocess`` argv (``diff_against``); an option-shaped
        run id (``--to=…``) would be parsed as a CLI flag downstream. Reject
        anything outside the strict ``[A-Za-z0-9_-]{1,64}`` allowlist and
        surface a clean error instead of storing it.
        """
        # Strip whitespace so a copy-pasted run id with trailing whitespace
        # doesn't trip RunHistoryManager's exact-id lookup downstream.
        cleaned, err = _validate_run_id(value)
        self.diff_other_run_id = cleaned
        if err:
            self.action_error = err
        elif self.action_error.startswith("Invalid run id"):
            # Clear a stale run-id validation error once the field is valid.
            self.action_error = ""

    @rx.event
    def diff_with_input(self) -> None:
        """Form submit handler — calls ``diff_against`` with the input value.

        Thin wrapper so the Compare button can be a plain ``on_click`` rather
        than needing to thread the input's local var through the closure.
        Mirrors the input-state-then-handler pattern Train/Multi-Run use for
        every config field.
        """
        # FRONTEND-B-002 (Stage C humanization): clear ``action_result`` on
        # both validation-failure branches. Pre-fix, a previous Replay /
        # Export success could sit next to a fresh Diff validation error,
        # rendering two contradictory action-panel messages at once. Mirrors
        # the OTHER-field-clear pattern already used in ``diff_against``.
        if not self.diff_other_run_id:
            self.action_error = "Enter a comparison run id."
            self.action_result = ""
            return
        if self.diff_other_run_id == self.current_run_id:
            self.action_error = (
                "Comparison run id must differ from the current run id."
            )
            self.action_result = ""
            return
        self.diff_against(self.diff_other_run_id)

    @rx.event
    def load_run(self) -> None:
        """Populate fields from the on-disk run history."""
        from pathlib import Path as _Path

        # Resolve run_id from the dynamic route parameter; fall back to
        # whatever the operator set programmatically into ``current_run_id``.
        # Route arg is named ``rid`` (not ``run_id``) to avoid shadowing the
        # existing ``AppState.run_id`` state var — Reflex 0.9 refuses to
        # bind a dynamic route arg that shadows any state var anywhere in
        # the state tree (DynamicRouteArgShadowsStateVarError).
        route_run_id = ""
        try:
            route_run_id = str(self.router.page.params.get("rid", "") or "")
        except Exception:  # noqa: BLE001 — defensive
            pass  # nosec B110 — defensive route-param read; missing param falls back to ""

        self.loading = True
        self.error = ""
        self.not_found = False
        # FRONTEND-B-001 (v1.4 Wave 3.5): clear the post-delete chrome
        # when (re)loading any run — navigating from a just-deleted run's
        # URL to a different run id must not carry the "Run deleted."
        # surface forward. This reset happens BEFORE the UI-A-003 route-param
        # validation early-return so a malformed URL still clears the chrome.
        self.was_deleted = False

        # UI-A-003 (Wave A1 HIGH): validate the route param at the trust
        # boundary before it reaches ``current_run_id`` (and thence the
        # diff/replay subprocess argv). An option-shaped ``rid`` (``--to=…``)
        # would be parsed as a CLI flag downstream. Reject it with a clean
        # error rather than assigning the malicious value.
        if route_run_id:
            cleaned_rid, rid_err = _validate_run_id(route_run_id)
            if rid_err:
                self.error = rid_err
                self.not_found = True
                self.loading = False
                return
            self.current_run_id = cleaned_rid

        try:
            try:
                from .ui_security import get_ui_output_dir
                history_dir = get_ui_output_dir()
            except Exception:
                history_dir = _Path.home() / ".backpropagate" / "ui-outputs"

            if not history_dir.exists():
                # V2-a (sibling): ``error`` is a public RunDetailState var;
                # history_dir embeds the home dir + username. Redact.
                self.error = _redact_action(f"No run history at {history_dir}.")
                return

            try:
                from .checkpoints import RunHistoryManager
            except ImportError as exc:
                self.error = _redact_action(f"checkpoints module unavailable: {exc}")
                return

            located_dir, job_dir = _locate_run_history_dir(history_dir, self.current_run_id)
            manager = RunHistoryManager(str(located_dir))
            entry = manager.get_run(self.current_run_id) if self.current_run_id else None
            if entry is None:
                self.not_found = True
                self.error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                return

            # Populate header fields. A UI job's own record knows a stop
            # (history says "completed" when training returned normally).
            self.status = (
                (_job_status_override(job_dir) if job_dir is not None else None)
                or str(entry.get("status") or "-")
            )
            self.model = str(entry.get("model_name") or "-")
            # Re-audit MEDIUM: dataset_info is often an absolute path (trainer
            # records the raw --data arg); redact home-dir/username before it
            # reaches this public, client-serialized var (sibling of UI-A-002).
            self.dataset = _dataset_label(entry.get("dataset_info"))
            self.started_at = _fmt_started(entry.get("started_at") or entry.get("timestamp"))
            self.completed_at = str(entry.get("completed_at") or "-")
            duration = entry.get("duration_seconds")
            if duration is None:
                self.duration = "-"
            else:
                try:
                    self.duration = f"{float(duration):.0f}s"
                except (TypeError, ValueError):
                    self.duration = "-"
            final_loss = entry.get("final_loss")
            if final_loss is None:
                self.final_loss = "-"
            else:
                try:
                    self.final_loss = f"{float(final_loss):.4f}"
                except (TypeError, ValueError):
                    self.final_loss = "-"
            # UI-A-002: full path into the backend-only var; the client reads
            # the redacted ``checkpoint_path_display`` computed var.
            self._checkpoint_path = str(entry.get("checkpoint_path") or "-")
            export_source = ""
            if job_dir is not None and (_Path(str(job_dir)) / "output").is_dir():
                export_source = str(_Path(str(job_dir)) / "output")
            elif entry.get("checkpoint_path") and _Path(
                str(entry["checkpoint_path"])
            ).expanduser().is_dir():
                export_source = str(_Path(str(entry["checkpoint_path"])).expanduser())
            self._export_source = export_source
            self.can_export_model = bool(export_source)

            # Hyperparameter table — flatten the entry dict into {key, value}
            # rows, skipping the fields surfaced as headers above so the
            # operator only sees the per-run config knobs.
            header_keys = {
                "run_id", "status", "model_name", "dataset_info",
                "started_at", "completed_at", "timestamp",
                "duration_seconds", "final_loss", "checkpoint_path",
                "loss_history",
            }
            hp_rows: list[dict] = []
            for key in sorted(entry.keys()):
                if key in header_keys:
                    continue
                value = entry[key]
                # Coerce all values to short strings for table render.
                if isinstance(value, (dict, list)):
                    import json as _json
                    value_str = _json.dumps(value, default=str)[:200]
                else:
                    value_str = str(value)[:200]
                hp_rows.append({"key": key, "value": value_str})
            self.hyperparameters = hp_rows

            # Loss history — embedded in the run entry (preferred) or
            # read from training_metrics.jsonl alongside the checkpoint.
            loss_hist = entry.get("loss_history") or []
            if isinstance(loss_hist, list):
                try:
                    self.loss_history = [float(x) for x in loss_hist if isinstance(x, (int, float))]
                except (TypeError, ValueError):
                    self.loss_history = []
            else:
                self.loss_history = []
            # UI job: the progress events hold the loss curve.
            if not self.loss_history and job_dir is not None:
                self.loss_history = _job_loss_curve(job_dir)

            # Checkpoint list — walk the checkpoint_path directory.
            self.checkpoints = []
            if entry.get("checkpoint_path"):
                cp_dir = _Path(str(entry["checkpoint_path"])).expanduser()
                if cp_dir.exists() and cp_dir.is_dir():
                    try:
                        for child in sorted(cp_dir.iterdir()):
                            if not child.is_dir():
                                continue
                            size_bytes = sum(
                                p.stat().st_size for p in child.rglob("*") if p.is_file()
                            )
                            self.checkpoints.append({
                                "name": child.name,
                                "size_mb": f"{size_bytes / (1024**2):.1f}",
                                "size_label": _fmt_bytes(size_bytes),
                                "timestamp": str(
                                    __import__("datetime").datetime.fromtimestamp(
                                        child.stat().st_mtime
                                    ).isoformat(timespec="seconds")
                                ),
                            })
                    except OSError:
                        # Best-effort — the run page still renders without
                        # checkpoints if filesystem walk fails.
                        pass

            # Log tail — read last 200 lines from training.log if present
            # alongside the checkpoint. Capped at 200 lines (~16 KB) to
            # keep the WS bundle small.
            self.log_lines = []
            # UI job: the child's stdout/stderr is <job>/output.log.
            log_candidates = []
            if job_dir is not None:
                log_candidates.append(_Path(str(job_dir)) / "output.log")
            if entry.get("checkpoint_path"):
                log_candidates.append(
                    _Path(str(entry["checkpoint_path"])).expanduser() / "training.log"
                )
            for log_path in log_candidates:
                if log_path.exists() and log_path.is_file():
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            tail = f.readlines()[-200:]
                            # UI-A-002: a training log routinely embeds absolute
                            # paths (checkpoint dir, HF cache, tempdirs) that
                            # carry the operator's home dir + username. Redact
                            # each line before it enters the client-serialized
                            # ``log_lines`` var. tqdm redraws with \r; keep
                            # each line's last frame.
                            self.log_lines = [
                                _redact_action(line.rstrip("\n").split("\r")[-1])
                                for line in tail
                            ]
                        break
                    except OSError:
                        pass
        finally:
            self.loading = False

    @rx.var
    def loss_chart_data(self) -> list[dict]:
        """Shape ``loss_history`` for ``rx.recharts.line_chart`` consumption.

        Returns ``[{"step": i, "loss": v}, ...]`` — the dict shape recharts
        needs. Computed-Var so the chart re-renders without an explicit
        event handler when ``load_run`` repopulates ``loss_history``.
        """
        return [{"step": i, "loss": v} for i, v in enumerate(self.loss_history)]

    @rx.event
    def diff_against(self, other_run_id: str) -> None:
        """Shell out to ``backprop diff-runs <self.current_run_id> <other_run_id>``.

        Bridge owns the subcommand (V1_3_BRIEF / Wave 6b BRIDGE-6); this
        UI just dispatches and surfaces the result. Failures surface via
        ``action_error``.
        """
        import shutil
        import subprocess

        if not self.current_run_id or not other_run_id:
            self.action_error = "Both run IDs are required for diff."
            return
        cmd = shutil.which("backprop") or shutil.which("backpropagate")
        if not cmd:
            self.action_error = "`backprop` CLI not found on PATH."
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): mark the shell-out
        # as in flight so the action panel renders an inline spinner. Cleared
        # in the finally so a thrown exception still leaves the UI in a
        # consistent state.
        # UI-A-003 (Wave A1 HIGH): validate both run IDs at the boundary even
        # though set_diff_other_run_id / load_run already guard their inputs —
        # belt-and-suspenders so a future caller that sets current_run_id /
        # other_run_id by another path can't smuggle an option-shaped id into
        # argv. Reject rather than shelling out.
        _, err_cur = _validate_run_id(self.current_run_id)
        _, err_other = _validate_run_id(other_run_id)
        if err_cur or err_other:
            self.action_error = err_cur or err_other
            return
        self.action_in_flight = "diff-runs"
        try:
            # ``--`` ends option parsing so the run IDs are always treated as
            # positionals by the downstream argparse CLI, never as flags.
            result = subprocess.run(
                [cmd, "diff-runs", "--", self.current_run_id, other_run_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                self.action_result = result.stdout[:5000]
                self.action_error = ""
            else:
                self.action_error = (result.stderr or result.stdout)[:1000]
                self.action_result = ""
        except (subprocess.TimeoutExpired, OSError) as exc:
            # V2-a (sibling): the OSError / TimeoutExpired repr can embed the
            # resolved ``backprop`` binary path (an absolute path under the
            # operator's environment); redact before surfacing to the client.
            self.action_error = _redact_action(f"diff-runs failed: {exc}")
        finally:
            self.action_in_flight = ""

    @rx.event
    def replay(self) -> None:
        """Validate that the current run is replayable IN-PROCESS.

        Pre-fix this shelled out to ``backprop replay --dry-run -- <id>`` —
        but the ``replay`` subcommand has NO ``--dry-run`` flag (it accepts
        only ``run_id`` / ``--output`` / ``--override`` / ``--json``), so
        argparse rejected the token (``unrecognized arguments: --dry-run``)
        and the Replay button always failed. This is the same phantom-CLI-
        surface bug class as UI-A-004 (delete_run / export_run) — the sibling
        fix the UI-A-003 separator test flagged as out-of-scope at the time —
        and the remedy matches UI-A-004: do the check IN-PROCESS via
        ``RunHistoryManager``. No subprocess, no PATH dependency (the
        ``backprop`` binary need not be on the Reflex server's PATH), no
        phantom flag.

        The button is a *preflight*, not the heavy replay: it confirms the
        run exists and carries the one hard precondition ``cmd_replay``
        enforces before launching training — a recorded ``dataset_info``
        (cli.py ``cmd_replay`` returns EXIT_USER_ERROR when it is ``None``) —
        then directs the operator to run ``backprop replay <id>`` from the
        shell to start the actual (heavy) job.
        """
        if not self.current_run_id:
            self.action_error = "No run loaded."
            return
        # UI-A-003: validate even though the id no longer reaches argv — a
        # malformed id should surface a clean error rather than a confusing
        # "not found" deeper in the lookup.
        _, rid_err = _validate_run_id(self.current_run_id)
        if rid_err:
            self.action_error = rid_err
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): see diff_against.
        self.action_in_flight = "replay"
        try:
            history_dir = self._resolve_history_dir()
            if not history_dir.exists():
                # V2-a: history_dir embeds the home dir + username; redact
                # before assigning to the client-serialized action_error.
                self.action_error = _redact_action(f"No run history at {history_dir}.")
                self.action_result = ""
                return
            from .checkpoints import RunHistoryManager

            manager = RunHistoryManager(
                str(_locate_run_history_dir(history_dir, self.current_run_id)[0])
            )
            entry = manager.get_run(self.current_run_id)
            if entry is None:
                self.action_error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                self.action_result = ""
                return
            # Mirror cmd_replay's one hard precondition (cli.py cmd_replay):
            # a run with no recorded dataset_info cannot be replayed
            # automatically, so a dry-run reporting "OK" here would promise a
            # replay the real command rejects. Surface the same verdict.
            if entry.get("dataset_info") is None:
                self.action_error = (
                    f"Run '{self.current_run_id}' has no dataset_info "
                    "recorded — cannot replay automatically. Re-run manually "
                    "with `backprop train --data <dataset>` matching the "
                    "original configuration."
                )
                self.action_result = ""
                return
            model = entry.get("model_name") or "(default model)"
            session_kind = entry.get("session_kind") or "single_run"
            self.action_result = (
                f"Dry-run OK — run {self.current_run_id} is replayable "
                f"(session={session_kind}, model={model}, "
                f"dataset={entry.get('dataset_info')}). To actually replay: "
                f"`backprop replay {self.current_run_id}` from the shell."
            )
            self.action_error = ""
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # V2-a: a raw OSError repr embeds absolute paths (the home dir +
            # username); redact before surfacing to the client-serialized var.
            self.action_error = _redact_action(
                f"replay check failed: {type(exc).__name__}: {exc}"
            )
            self.action_result = ""
        finally:
            self.action_in_flight = ""

    def _resolve_history_dir(self):
        """Resolve the sandboxed run-history directory.

        Mirrors ``load_run``'s resolution: prefer the UI sandbox
        (``get_ui_output_dir()``), fall back to the legacy default. Kept as
        one helper so the in-process action handlers (UI-A-004) and
        ``load_run`` stay in lockstep.
        """
        from pathlib import Path as _Path

        try:
            from .ui_security import get_ui_output_dir

            return get_ui_output_dir()
        except Exception:  # noqa: BLE001 — defensive; fall back to legacy default
            return _Path.home() / ".backpropagate" / "ui-outputs"

    @rx.event
    def delete_run(self) -> None:
        """Delete the current run's history entry IN-PROCESS (UI-A-004).

        Pre-fix this shelled out to ``backprop delete-run <id> --yes`` — a
        subcommand that DOES NOT EXIST (argparse rejected it → the Delete
        button always failed). We use ``RunHistoryManager.delete_run``
        directly, exactly as ``load_run`` reads via ``get_run``. No
        subprocess, no PATH dependency, no phantom CLI surface.

        Operator confirmation is the responsibility of the UI button (a
        confirm-dialog wrap); this handler unconditionally executes.
        """
        if not self.current_run_id:
            self.action_error = "No run loaded."
            return
        # UI-A-003: the id flows nowhere dangerous now (in-process), but
        # validate anyway so a malformed id surfaces a clean error rather
        # than a confusing "not found".
        _, rid_err = _validate_run_id(self.current_run_id)
        if rid_err:
            self.action_error = rid_err
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): see diff_against.
        self.action_in_flight = "delete-run"
        try:
            history_dir = self._resolve_history_dir()
            if not history_dir.exists():
                # V2-a: redact the home-dir-bearing history_dir before it
                # reaches the client-serialized action_error.
                self.action_error = _redact_action(f"No run history at {history_dir}.")
                self.action_result = ""
                return
            from .checkpoints import RunHistoryManager

            manager = RunHistoryManager(
                str(_locate_run_history_dir(history_dir, self.current_run_id)[0])
            )
            deleted = manager.delete_run(self.current_run_id)
            if deleted:
                self.action_result = f"Run {self.current_run_id} deleted."
                self.action_error = ""
                # FRONTEND-B-001 (v1.4 Wave 3.5): a successful delete must
                # render the "Run deleted." chrome with a Back-to-runs
                # affordance, NOT the "Run not found" chrome. Setting
                # ``not_found = True`` here pre-fix latched the page into
                # the unknown-id surface and stranded ``action_result``
                # behind a template that never displays it. The template
                # in ``run_detail.py`` branches on ``was_deleted`` BEFORE
                # ``not_found`` so the deletion confirmation wins.
                self.was_deleted = True
            else:
                self.action_error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                self.action_result = ""
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # V2-a: redact absolute paths from the raw exception repr.
            self.action_error = _redact_action(
                f"delete failed: {type(exc).__name__}: {exc}"
            )
            self.action_result = ""
        finally:
            self.action_in_flight = ""

    @rx.event
    def export_run(self) -> None:
        """Export the current run as a single-record JSONL IN-PROCESS (UI-A-004).

        Pre-fix this shelled out to ``backprop export-runs --run-id <id>`` —
        but ``export-runs`` has NO ``--run-id`` flag (only ``--output/-o``,
        ``--format``, ``--to``, ``--status``), so argparse rejected it and
        the Export button always failed. We read the single run via
        ``RunHistoryManager.get_run`` and write one JSONL record to a
        SANDBOXED path under ``get_ui_output_dir()`` — no phantom flag, no
        filesystem-wide write surface.
        """
        import json
        from datetime import datetime, timezone

        if not self.current_run_id:
            self.action_error = "No run loaded."
            return
        _, rid_err = _validate_run_id(self.current_run_id)
        if rid_err:
            self.action_error = rid_err
            return
        # FRONTEND-B-014-EXTENDED (Stage C humanization): see diff_against.
        self.action_in_flight = "export-runs"
        try:
            history_dir = self._resolve_history_dir()
            if not history_dir.exists():
                # V2-a: redact the home-dir-bearing history_dir.
                self.action_error = _redact_action(f"No run history at {history_dir}.")
                self.action_result = ""
                return
            from .checkpoints import RunHistoryManager

            manager = RunHistoryManager(
                str(_locate_run_history_dir(history_dir, self.current_run_id)[0])
            )
            entry = manager.get_run(self.current_run_id)
            if entry is None:
                self.action_error = _redact_action(
                    f"Run '{self.current_run_id}' not found in {history_dir}."
                )
                self.action_result = ""
                return

            # Write inside a sandboxed ``exports/`` subdir of the UI output
            # dir. The filename is derived from the (already-validated, so
            # filesystem-safe ``[A-Za-z0-9_-]``) run id plus a UTC stamp so
            # repeat exports don't clobber. No user-controlled path segment
            # escapes the sandbox.
            exports_dir = history_dir / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = exports_dir / f"run-{self.current_run_id}-{stamp}.jsonl"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry, default=str))
                fh.write("\n")
            # V2-a: out_path is the sandbox absolute path (home dir +
            # username); redact before surfacing the success string to the
            # client. The basename is preserved so the operator can still
            # locate the file inside their UI output dir.
            self.action_result = _redact_action(
                f"Exported run {self.current_run_id} to {out_path}."
            )
            self.action_error = ""
        except Exception as exc:  # noqa: BLE001 — operator-facing string
            # V2-a: redact absolute paths from the raw exception repr.
            self.action_error = _redact_action(
                f"export failed: {type(exc).__name__}: {exc}"
            )
            self.action_result = ""
        finally:
            self.action_in_flight = ""

    @rx.event
    def export_model(self):
        """Open the Export page with this run's adapter as the source (P2)."""
        if not self._export_source:
            self.action_error = "This run has no saved adapter folder to export."
            return None
        return [
            ExportState.set_source_model_path(self._export_source),
            rx.redirect("/export"),
        ]

    @rx.event
    def clear_action_message(self) -> None:
        """Dismiss the action result / error banner."""
        self.action_result = ""
        self.action_error = ""
        self.action_error_suggestion = ""


# ---------------------------------------------------------------------------
# ModelsState — backs /models (Wave 6b)
# ---------------------------------------------------------------------------
#
# Lists local Hugging Face cache contents + per-model disk usage + unused-
# model cleanup affordance. Pulls from ``~/.cache/huggingface/hub/``
# directly via filesystem APIs — no `huggingface_hub` dep required (avoids
# pulling the heavy optional dep into the [ui] extra path).
#
# v1.3 surfaces:
#   - Total cache size + per-model breakdown
#   - Last-modified timestamp (proxy for "last used")
#   - Delete-model affordance (operator confirms before the rm -rf)


def _hf_hub_cache_dir():
    """The Hugging Face hub cache this machine uses (ui-v2 P2).

    Resolved at call time in ``huggingface_hub``'s own order:
    ``HF_HUB_CACHE`` (or the older ``HUGGINGFACE_HUB_CACHE``), then
    ``HF_HOME/hub``, then ``XDG_CACHE_HOME/huggingface/hub``, then
    ``~/.cache/huggingface/hub``. Hard-coding the last one showed the wrong,
    often near-empty folder on machines that move the cache to a bigger
    drive.
    """
    import os as _os
    from pathlib import Path as _Path

    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = _os.environ.get(var, "").strip()
        if value:
            return _Path(value).expanduser()
    hf_home = _os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return _Path(hf_home).expanduser() / "hub"
    xdg = _os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return _Path(xdg).expanduser() / "huggingface" / "hub"
    return _Path.home() / ".cache" / "huggingface" / "hub"


def _home_relative(path: str) -> str:
    """``~/...`` for a path under the home directory (hides the username
    without the literal ``<redacted-path>`` showing up on screen); other
    paths unchanged."""
    from pathlib import Path as _Path

    try:
        rel = _Path(path).resolve().relative_to(_Path.home().resolve())
    except (OSError, ValueError):
        return path
    return "~/" + rel.as_posix() if str(rel) != "." else "~"


class ModelsState(rx.State):
    """Models surface state — local HF cache inventory."""

    models: list[dict] = []
    total_size_label: str = ""
    # UI-A-002 (Wave A2): the HF cache dir is ``~/.cache/huggingface/hub`` —
    # it embeds the operator's home dir + username. Held in a backend-only
    # var; the client reads the redacted ``cache_dir_display`` computed var.
    _cache_dir: str = ""
    loading: bool = False
    error: str = ""
    # HUX-02 (Stage C humanization): operator-actionable remedy, held
    # separately from ``error`` so the /models callout can render it on its own
    # dimmed ``hint=`` line. ModelsState errors are currently all path-redacted
    # informational strings (no ``sanitize_error_for_user`` suggestion split),
    # so this stays empty today — the slot is wired backward-compatibly so a
    # future suggestion-bearing error surfaces in the same scannable hierarchy
    # as /runs and /run-detail.
    error_suggestion: str = ""
    last_loaded_at: str = ""
    last_loaded_label: str = ""
    # CLIUI-B-005 (Stage C): per-row in-flight flag for the delete affordance.
    # Holds the ``dir_name`` currently being deleted (empty when idle). Drives
    # the row's ``disabled=`` binding AND gates re-entry inside delete_model so
    # a double-click can't fire a second rmtree that finds the dir already gone
    # and surfaces a spurious "not found" error. Mirrors
    # RunDetailState.action_in_flight.
    deleting_dir: str = ""

    @rx.var
    def cache_dir_display(self) -> str:
        """Client-facing, redacted form of the HF cache directory (UI-A-002)."""
        if not self._cache_dir:
            return ""
        # UI-A-002: no username in the UI — a home-relative "~/..." rather
        # than the literal "<redacted-path>" prefix (ui-v2 P2).
        return _home_relative(self._cache_dir)

    @rx.event
    def load_models(self) -> None:
        """Walk the HF hub cache (``_hf_hub_cache_dir``) and fill ``self.models``.

        HF cache layout (per huggingface_hub docs):
            <cache>/models--<owner>--<model>/snapshots/<sha>/<files>
            <cache>/models--<owner>--<model>/refs/<rev>

        We surface one entry per ``models--*`` top-level dir; size is the
        recursive sum of all files inside.
        """
        from datetime import datetime, timezone

        self.loading = True
        self.error = ""
        try:
            cache_dir = _hf_hub_cache_dir()
            # UI-A-002: full path into the backend-only var; the client reads
            # the redacted ``cache_dir_display`` computed var.
            self._cache_dir = str(cache_dir)
            if not cache_dir.exists():
                self.models = []
                self.total_size_label = ""
                # UI-A-002: ``error`` is a public var; redact the cache path.
                self.error = (
                    f"No Hugging Face cache at {_home_relative(str(cache_dir))} yet. "
                    "Models are downloaded there the first time a run uses them."
                )
                return

            model_rows: list[dict] = []
            total_bytes = 0
            try:
                for entry in sorted(cache_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    name = entry.name
                    if not name.startswith("models--"):
                        continue
                    # Unmangle: ``models--meta-llama--Llama-3.1-8B`` ->
                    # ``meta-llama/Llama-3.1-8B``
                    pretty = name[len("models--"):].replace("--", "/", 1)
                    try:
                        size_bytes = sum(
                            p.stat().st_size for p in entry.rglob("*") if p.is_file()
                        )
                    except OSError:
                        size_bytes = 0
                    total_bytes += size_bytes
                    mtime = 0.0
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        pass
                    model_rows.append({
                        "name": pretty,
                        "dir_name": name,
                        "size_mb": f"{size_bytes / (1024**2):.1f}",
                        "size_label": _fmt_bytes(size_bytes),
                        "size_bytes": size_bytes,
                        "last_modified": (
                            datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                            if mtime else "-"
                        ),
                    })
            except OSError as exc:
                # UI-A-002: the OSError repr embeds the cache path.
                self.error = _redact_action(f"Cannot walk HF cache: {exc}")
                self.models = []
                self.total_size_label = ""
                return

            # Sort by size descending — heaviest cache offenders first.
            model_rows.sort(key=lambda r: r["size_bytes"], reverse=True)
            self.models = model_rows
            self.total_size_label = _fmt_bytes(total_bytes)
            self.last_loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.last_loaded_label = _fmt_local_time()
        finally:
            self.loading = False

    @rx.event
    def delete_model(self, dir_name: str) -> None:
        """Delete one ``models--*`` directory from the HF cache.

        The operator confirms via a UI button click; the handler unconditionally
        proceeds. Failures surface via ``self.error``.

        FRONTEND-F-007 safety: the path is validated to live under
        ``~/.cache/huggingface/hub/`` so a malicious operator-controlled
        ``dir_name`` can't escape the cache via ``..`` traversal.

        CLIUI-B-005 (Stage C) re-entrancy guard: the delete affordance had no
        in-flight disable, so a double-click re-entered this handler; the
        second invocation found the directory already gone and surfaced a
        spurious "Model directory not found" error. The handler now records the
        in-flight ``dir_name`` in ``self.deleting_dir`` and SHORT-CIRCUITS a
        re-entrant click for the same target (a no-op, not an error). The flag
        is cleared via try/finally so a thrown ``OSError`` can't latch the
        button disabled forever. The page also binds ``disabled`` to this flag,
        but the handler-level guard makes the behaviour correct even if a
        rapid double-fire slips past the UI debounce.

        UI-A-006 (Wave A2) hardening:
          - Use ``Path.is_relative_to`` for the confinement check instead of
            ``str.startswith``. A string-prefix test treats a SIBLING dir
            whose name shares the cache prefix (e.g. ``…/hub-evil``) as
            "inside" the cache; the path-component check does not.
          - Refuse the delete if ``target`` (the UNRESOLVED path) is a
            symlink. Pre-fix the handler resolved the symlink and then
            ``rmtree``'d the resolved target — a ``models--*`` symlink in the
            cache pointing at, say, ``~/important`` would pass the
            resolved-prefix check (target resolves outside, but the OLD
            ``startswith`` compared against the resolved cache root) OR, worse,
            delete the link's target outside the cache. Refusing symlinks
            outright removes the foot-gun; real HF cache entries are plain
            directories.
        """
        import shutil

        cache_dir = _hf_hub_cache_dir()
        if not dir_name or not dir_name.startswith("models--") or "/" in dir_name or "\\" in dir_name or ".." in dir_name:
            self.error = f"Invalid model directory name: {dir_name!r}"
            return
        # CLIUI-B-005: short-circuit a re-entrant double-click for the same
        # target. The first click is still in flight (its reload hasn't yet
        # refreshed the row away), so a second fire would rmtree an
        # already-deleted dir and surface a confusing "not found". Treat it as
        # a silent no-op — the in-flight delete will finish and reload.
        if self.deleting_dir == dir_name:
            return
        target = cache_dir / dir_name
        self.deleting_dir = dir_name
        try:
            # UI-A-006: refuse symlinks BEFORE resolving so a malicious
            # ``models--*`` symlink can't redirect the rmtree outside the
            # cache. Check the unresolved path.
            if target.is_symlink():
                self.error = _redact_action(
                    f"Refusing to delete a symlinked cache entry: {target}"
                )
                return
            target_resolved = target.resolve()
            cache_resolved = cache_dir.resolve()
            # UI-A-006: path-component confinement (is_relative_to), not a
            # string-prefix test. ``is_relative_to`` returns False for a
            # sibling dir that merely shares the prefix string.
            if not target_resolved.is_relative_to(cache_resolved):
                self.error = _redact_action(
                    f"Refusing to delete outside HF cache: {target_resolved}"
                )
                return
            if not target_resolved.exists():
                self.error = _redact_action(f"Model directory not found: {target}")
                return
            # Operate on the confined, resolved path. ``rmtree`` does not
            # follow a top-level symlink (and we refused one above anyway).
            shutil.rmtree(target_resolved)
        except OSError as exc:
            self.error = _redact_action(f"Failed to delete {dir_name}: {exc}")
            return
        finally:
            # CLIUI-B-005: clear the in-flight flag on EVERY exit path
            # (success, validation refusal, or thrown OSError) so the row's
            # delete button re-enables.
            self.deleting_dir = ""
        # Reload to reflect the deletion.
        self.load_models()


__all__ = [
    "RunState",
    "Theme",
    "ActiveSurface",
    "ExportFormat",
    "Quantization",
    "MergeMode",
    "GgufQuant",
    "DatasetFormatHint",
    "AppState",
    "TrainState",
    "MultiRunState",
    "ExportState",
    "DatasetState",
    "RunsState",
    "RunDetailState",
    "ModelsState",
    "AuthBadgeState",
]
