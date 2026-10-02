"""UI job manager — the parent (UI-server) half of the ui-v2 job contract.

ui-v2 P1. Training NEVER happens in the UI server process: the manager spawns
``python -m backpropagate train --ui-run-dir <run_dir>`` as a child and
observes it through files (``events.jsonl`` / ``control.json`` / ``job.json``
/ ``output.log`` — see :mod:`backpropagate.job_events` for the schema). No
pipes: a UI crash can never wedge on a dead pipe buffer, and a tab reload
reattaches by reading the same files.

Process guarantees (handoff "Process and control contract"):

- One job at a time — a second :meth:`start` is refused ON SCREEN (the caller
  surfaces ``JobRefusedError.message``).
- Crash lease: on Windows the child is assigned to a Job Object with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` so the whole tree dies with the UI
  server; on POSIX the child runs in its own session (``start_new_session``)
  and teardown uses the existing ``backpropagate.cli._kill_process_tree``.
- Stop is "Stop and save checkpoint": write ``control.json``, wait a grace
  window (``max(60s, 3 x last step time + 20s)``), then escalate to the
  tree kill. Never CTRL_* / POSIX signal tricks (SIGABRT on Windows is a
  hard kill).

This module is Reflex-free so it is unit-testable with fake processes and so
the Store-side smoke harness can drive it headless.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .job_events import (
    CONTROL_FILENAME,
    EVENTS_FILENAME,
    JobEventWriter,
)

logger = logging.getLogger(__name__)

JOB_FILENAME = "job.json"
#: How long a job that wrote its terminal event may keep reading "active"
#: while its process is still exiting.
EXIT_GRACE_S = 120.0
OUTPUT_LOG = "output.log"

#: Server-side caps (handoff rule 7). The UI form enforces tighter values;
#: these are the last rail against a crafted websocket frame.
MAX_UI_STEPS = 100_000
MAX_UI_BATCH = 256
MAX_UI_SEQ_LENGTH = 131_072
MAX_UI_LORA_R = 512
MAX_UI_LR = 1.0
MAX_UI_RUNS = 50
MAX_UI_SAMPLES = 1_000_000

#: Export formats / GGUF levels the UI may request: exactly the CLI's
#: ``backprop export --format`` / ``--quantization`` choices.
UI_EXPORT_FORMATS = ("lora", "merged", "gguf")
UI_GGUF_QUANTS = ("f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k")
#: Multi-run merge choices the UI offers, mapped to CLI flags in the argv.
UI_MERGE_CHOICES = ("slao", "simple", "ties")
#: Training objectives (``--method``) and the per-method knobs the UI may
#: send, keyed to their CLI flag and allowed range (ui-v2 P3).
UI_METHODS = ("sft", "orpo", "simpo", "kto")
UI_METHOD_PARAMS: dict[str, dict[str, tuple[str, float, float]]] = {
    "sft": {},
    "orpo": {"orpo_beta": ("--orpo-beta", 0.0, 10.0)},
    "simpo": {
        "simpo_beta": ("--simpo-beta", 0.0, 50.0),
        "simpo_gamma": ("--simpo-gamma", 0.0, 50.0),
    },
    "kto": {
        "kto_beta": ("--kto-beta", 0.0, 10.0),
        "kto_desirable_weight": ("--kto-desirable-weight", 0.0, 100.0),
        "kto_undesirable_weight": ("--kto-undesirable-weight", 0.0, 100.0),
    },
}
#: ``--gpu-max-temp`` range the CLI accepts (°C).
UI_GPU_TEMP_RANGE = (50.0, 105.0)
MAX_UI_LORA_ALPHA = 1024
_TARGET_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_OLLAMA_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

#: Cooperative-stop grace floor; the live window is
#: ``max(STOP_GRACE_FLOOR_S, 3 * last_step_s + 20)``.
STOP_GRACE_FLOOR_S = 60.0

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$")

#: Auth secrets of the UI server that a job child has no use for.
_CHILD_ENV_DROP_PREFIXES = ("BACKPROPAGATE_UI_AUTH",)
_CHILD_ENV_DROP = frozenset({"BACKPROPAGATE_UI_LAUNCH_TOKEN"})

#: Bytes of events.jsonl read per poll, and the tail window that holds the
#: terminal event.
MAX_EVENT_READ_BYTES = 4 * 1024 * 1024
_EVENT_TAIL_BYTES = 256 * 1024

_INT_FIELDS = ("steps", "lora_r", "runs")
_OPTIONAL_INT_FIELDS = ("samples", "lora_alpha")
_OPTIONAL_REAL_FIELDS = ("lora_dropout", "gpu_max_temp")
_BOOL_FIELDS = ("base_4bit", "gradient_checkpointing", "trust_remote_code")
#: String fields and their length caps. None of them may start with "-":
#: every one ends up as an argv value.
_STR_FIELDS: dict[str, int] = {
    "kind": 32, "model": 256, "dataset_path": 4096, "batch": 16, "mode": 32,
    "merge": 32, "source_path": 4096, "export_format": 32, "quantization": 32,
    "ollama_name": 128, "target_modules": 1024, "method": 32, "run_name": 128,
}


def _short(value: Any) -> str:
    """A bounded repr for error messages (a client controls the value)."""
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _require_types(spec: JobSpec) -> None:
    """Every field has the type the argv builders assume.

    The spec can come from a browser payload, where an int field may arrive
    as a bool, a list or a dict. Refuse here, with a bounded message, rather
    than fail inside ``int()`` or emit ``True`` as an argv value.
    """

    def is_int(v: Any) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def is_real(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    for name in _INT_FIELDS:
        if not is_int(getattr(spec, name)):
            raise JobValidationError(f"{name} must be a whole number (got {_short(getattr(spec, name))}).")
    for name in _OPTIONAL_INT_FIELDS:
        value = getattr(spec, name)
        if value is not None and not is_int(value):
            raise JobValidationError(f"{name} must be a whole number (got {_short(value)}).")
    if not is_real(spec.lr):
        raise JobValidationError(f"lr must be a number (got {_short(spec.lr)}).")
    for name in _OPTIONAL_REAL_FIELDS:
        value = getattr(spec, name)
        if value is not None and not is_real(value):
            raise JobValidationError(f"{name} must be a number (got {_short(value)}).")
    for name in _BOOL_FIELDS:
        if not isinstance(getattr(spec, name), bool):
            raise JobValidationError(f"{name} must be true or false (got {_short(getattr(spec, name))}).")
    for name, cap in _STR_FIELDS.items():
        value = getattr(spec, name)
        if not isinstance(value, str):
            raise JobValidationError(f"{name} must be text (got {_short(value)}).")
        if len(value) > cap:
            raise JobValidationError(f"{name} is too long ({len(value)} characters; the limit is {cap}).")
        if value.strip().startswith("-"):
            raise JobValidationError(f"{name} cannot start with '-' (got {_short(value)}).")
    for name in ("scratch_root", "output_dir"):
        value = getattr(spec, name)
        if value is not None and not isinstance(value, str):
            raise JobValidationError(f"{name} must be text (got {_short(value)}).")
    params = spec.method_params
    if not isinstance(params, dict) or len(params) > 16:
        raise JobValidationError(f"method_params must be a small mapping (got {_short(params)}).")
    for key, value in params.items():
        if not isinstance(key, str) or not is_real(value):
            raise JobValidationError(f"method_params must map names to numbers (got {_short({key: value})}).")


def _hf_cache_root() -> Path | None:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE).expanduser().resolve()
    except Exception:  # noqa: BLE001 - hub not installed / odd config
        return None


def _validate_model(model_text: str) -> str:
    """A Hugging Face id (``org/name``), or a local folder inside the UI
    output sandbox or the Hugging Face cache. Returns the value to put in
    argv: the id, or the RESOLVED local path.

    Before this, any existing local path was accepted, so a remote client
    could point a job at any directory the server can read.
    """
    model = (model_text or "").strip()
    if not model or len(model) > 256:
        raise JobValidationError("Model id is empty or overlong.")
    if _MODEL_ID_RE.match(model):
        return model
    path = Path(model).expanduser()
    if not path.exists():
        raise JobValidationError(
            f"Model {_short(model)} is neither a HuggingFace id (org/name) nor an "
            "existing local path."
        )
    try:
        from .ui_security import get_ui_output_dir

        resolved = path.resolve()
        roots = [Path(get_ui_output_dir()).resolve()]
    except Exception as exc:
        raise JobValidationError(
            f"Could not verify the UI sandbox for the model path; refusing to start ({exc!r})."
        ) from exc
    cache = _hf_cache_root()
    if cache is not None:
        roots.append(cache)
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise JobValidationError(
            "A local model must be inside the UI output folder or the Hugging Face "
            f"cache (got {_short(str(resolved))})."
        )
    return str(resolved)


class JobValidationError(ValueError):
    """Raised when a JobSpec fails server-side validation. Message is
    operator-safe and shown on screen."""


class JobRefusedError(RuntimeError):
    """Raised when a second job is started while one is active."""


@dataclass
class JobSpec:
    """Everything needed to build the child argv.

    Only knobs the ``backprop train`` / ``multi-run`` CLI actually exposes
    are carried here: every field maps to a real flag (ui-v2 P3 CLI parity),
    nothing is smuggled to the child through the back door.
    """

    kind: str  # "sft" (P1); "multi_run" / "export" (P2)
    model: str = ""
    dataset_path: str = ""
    steps: int = 100
    batch: str = "auto"  # "auto" or a positive-int string
    lr: float = 2e-4
    lora_r: int = 256
    mode: str = "lora"  # --mode lora|full (full: single run, SFT only)
    samples: int | None = None
    # ---- training knobs (P3 CLI parity; None/"" = the CLI default) -----
    lora_alpha: int | None = None  # --lora-alpha
    lora_dropout: float | None = None  # --lora-dropout
    target_modules: str = ""  # --target-modules ("q_proj,v_proj" | "all-linear")
    base_4bit: bool = True  # False -> --no-4bit (16-bit LoRA base)
    method: str = "sft"  # --method, one of UI_METHODS
    method_params: dict[str, float] = field(default_factory=dict)  # --orpo-beta ...
    run_name: str = ""  # --run-name
    gpu_max_temp: float | None = None  # --gpu-max-temp
    gradient_checkpointing: bool = True  # False -> --no-gradient-checkpointing
    scratch_root: str | None = None  # override for tests; default sandboxed
    output_dir: str | None = None  # default: <run_dir>/output
    trust_remote_code: bool = False
    # ---- multi_run (P2) ------------------------------------------------
    runs: int = 3
    merge: str = "slao"  # one of UI_MERGE_CHOICES
    # ---- export (P2) ---------------------------------------------------
    source_path: str = ""  # adapter / model dir, inside the UI sandbox
    export_format: str = "lora"  # one of UI_EXPORT_FORMATS
    quantization: str = "q4_k_m"  # one of UI_GGUF_QUANTS (gguf only)
    ollama_name: str = ""  # non-empty: register with Ollama (gguf only)


_ABS_PATH_RE = re.compile(r"^([A-Za-z]:[\\/]|\\\\|/)")


def _redact_argv(argv_tail: list[str]) -> list[str]:
    """Redact absolute-path VALUES from the job.json spawn record.

    job.json survives the run and can end up in screenshots/issues, so it
    must not leak local layout (dataset/output paths). Flags survive intact
    (they matter for reproduction); only values that parse as absolute
    filesystem paths collapse to ``<path>``. Replaces the rig-specific
    ``startswith("E:")`` filter from the first P1 cut.
    """
    out: list[str] = []
    for arg in argv_tail:
        if _ABS_PATH_RE.match(arg):
            out.append("<path>")
        else:
            out.append(arg)
    return out


@dataclass
class JobHandle:
    job_id: str
    run_dir: Path
    pid: int
    started_at: float
    spec: JobSpec = field(repr=False)

    @property
    def events_path(self) -> Path:
        return self.run_dir / EVENTS_FILENAME

    @property
    def job_path(self) -> Path:
        return self.run_dir / JOB_FILENAME

    @property
    def log_path(self) -> Path:
        return self.run_dir / OUTPUT_LOG


def _default_jobs_root() -> Path:
    """Scratch root for UI jobs. Lives inside the UI output sandbox so the
    Runs views and the sandbox rules cover it (``~/.backpropagate/ui-outputs/
    jobs``)."""
    try:
        from .ui_security import get_ui_output_dir

        return Path(get_ui_output_dir()) / "jobs"
    except Exception:  # noqa: BLE001 — ui_security unavailable (minimal install)
        return Path.home() / ".backpropagate" / "ui-outputs" / "jobs"


def _mint_job_id(clock: Callable[[], float]) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(clock()))
    import secrets

    return f"run_{stamp}_{os.getpid()}_{secrets.token_hex(2)}"


def _build_train_argv(spec: JobSpec, run_dir: Path) -> list[str]:
    output_dir = Path(spec.output_dir) if spec.output_dir else run_dir / "output"
    argv = [
        sys.executable,
        "-m",
        "backpropagate",
        "train",
        "--model",
        spec.model,
        "--data",
        spec.dataset_path,
        "--steps",
        str(spec.steps),
        "--lr",
        str(spec.lr),
        "--lora-r",
        str(spec.lora_r),
        "--mode",
        spec.mode,
        "--output",
        str(output_dir),
        "--ui-run-dir",
        str(run_dir),
    ]
    if spec.batch != "auto":
        argv += ["--batch-size", str(spec.batch)]
    if spec.samples:
        argv += ["--samples", str(spec.samples)]
    return argv + _training_flags(spec)


def _training_flags(spec: JobSpec) -> list[str]:
    """The P3 CLI-parity flags shared by ``train`` and ``multi-run``.

    Each is passed only when it differs from the CLI default, so the argv
    (and the job.json record) stays short and a default form reproduces a
    default CLI run exactly.
    """
    argv: list[str] = []
    if spec.mode == "lora":
        if spec.lora_alpha is not None:
            argv += ["--lora-alpha", str(int(spec.lora_alpha))]
        if spec.lora_dropout is not None:
            argv += ["--lora-dropout", f"{float(spec.lora_dropout):g}"]
        if spec.target_modules:
            argv += ["--target-modules", spec.target_modules]
        if not spec.base_4bit:
            argv += ["--no-4bit"]
    if spec.method != "sft":
        argv += ["--method", spec.method]
        for key, (flag, _lo, _hi) in UI_METHOD_PARAMS[spec.method].items():
            if key in spec.method_params:
                argv += [flag, f"{float(spec.method_params[key]):g}"]
    if spec.run_name:
        argv += ["--run-name", spec.run_name]
    if spec.gpu_max_temp is not None:
        argv += ["--gpu-max-temp", f"{float(spec.gpu_max_temp):g}"]
    if not spec.gradient_checkpointing and spec.mode == "lora":
        argv += ["--no-gradient-checkpointing"]
    return argv


def _build_multi_run_argv(spec: JobSpec, run_dir: Path) -> list[str]:
    output_dir = Path(spec.output_dir) if spec.output_dir else run_dir / "output"
    argv = [
        sys.executable,
        "-m",
        "backpropagate",
        "multi-run",
        "--model",
        spec.model,
        "--data",
        spec.dataset_path,
        "--runs",
        str(spec.runs),
        "--steps",
        str(spec.steps),
        "--mode",
        spec.mode,
        "--output",
        str(output_dir),
        "--ui-run-dir",
        str(run_dir),
    ]
    if spec.samples:
        argv += ["--samples", str(spec.samples)]
    argv += ["--lr", str(spec.lr), "--lora-r", str(spec.lora_r)]
    if spec.batch != "auto":
        argv += ["--batch-size", str(spec.batch)]
    if spec.merge == "simple":
        argv += ["--merge-mode", "simple"]
    else:
        argv += ["--merge-mode", "slao"]
        if spec.merge == "ties":
            argv += ["--merge-strategy", "ties"]
    return argv + _training_flags(spec)


def _build_export_argv(spec: JobSpec, run_dir: Path) -> list[str]:
    output_dir = Path(spec.output_dir) if spec.output_dir else run_dir / "output"
    argv = [
        sys.executable,
        "-m",
        "backpropagate",
        "export",
        "--format",
        spec.export_format,
        "--output",
        str(output_dir),
        "--ui-run-dir",
        str(run_dir),
    ]
    if spec.export_format == "gguf":
        argv += ["--quantization", spec.quantization]
        if spec.ollama_name:
            argv += ["--ollama", "--ollama-name", spec.ollama_name]
    # The positional goes last, after "--": it can never be read as a flag.
    return [*argv, "--", spec.source_path]


def _build_calibrate_argv(spec: JobSpec, run_dir: Path) -> list[str]:
    """``backprop estimate-vram <model> --calibrate``: measure on this GPU."""
    argv = [
        sys.executable,
        "-m",
        "backpropagate",
        "estimate-vram",
        "--calibrate",
        "--mode",
        spec.mode,
        "--ui-run-dir",
        str(run_dir),
    ]
    if spec.mode == "lora" and not spec.base_4bit:
        argv += ["--no-4bit"]
    # The positional goes last, after "--": it can never be read as a flag.
    return [*argv, "--", spec.model]


def _build_argv(spec: JobSpec, run_dir: Path) -> list[str]:
    if spec.kind == "multi_run":
        return _build_multi_run_argv(spec, run_dir)
    if spec.kind == "export":
        return _build_export_argv(spec, run_dir)
    if spec.kind == "calibrate":
        return _build_calibrate_argv(spec, run_dir)
    return _build_train_argv(spec, run_dir)


def _check_in_sandbox(path_text: str, what: str) -> Path:
    """Resolve ``path_text`` and require it inside the UI output sandbox
    (``get_ui_output_dir()``). Fails CLOSED when the sandbox cannot be
    verified (P1 fix-round review)."""
    path = Path(path_text).expanduser()
    if not path.exists():
        raise JobValidationError(f"{what} not found: {path}")
    try:
        from .ui_security import get_ui_output_dir

        base = Path(get_ui_output_dir()).resolve()
        resolved = path.resolve()
        if not (str(resolved) + os.sep).startswith(str(base) + os.sep) and resolved != base:
            raise JobValidationError(
                f"The UI only reads files inside {base} ({what.lower()}: {resolved})."
            )
    except JobValidationError:
        raise
    except Exception as exc:
        raise JobValidationError(
            f"Could not verify the UI sandbox for the {what.lower()}; refusing "
            f"to start ({exc!r})."
        ) from exc
    return resolved


#: JobSpec fields a browser client must never set: where the job writes and
#: whether model code runs. ``TrainState.start_job`` drops them, and
#: ``_validate_spec`` refuses a spec that points them outside the sandbox.
SERVER_ONLY_SPEC_FIELDS = frozenset({"scratch_root", "output_dir", "trust_remote_code"})


def _require_inside_sandbox(path_text: str, what: str) -> None:
    """Require ``path_text`` (which need not exist yet) to resolve inside the
    UI output sandbox. Fails CLOSED when the sandbox cannot be verified."""
    try:
        from .ui_security import get_ui_output_dir

        base = Path(get_ui_output_dir()).resolve()
        resolved = Path(path_text).expanduser().resolve()
    except Exception as exc:
        raise JobValidationError(
            f"Could not verify the UI sandbox for the {what.lower()}; refusing "
            f"to start ({exc!r})."
        ) from exc
    if resolved != base and base not in resolved.parents:
        raise JobValidationError(f"The UI only writes inside {base} ({what.lower()}: {resolved}).")


def _validate_export_spec(spec: JobSpec) -> None:
    source = (spec.source_path or "").strip()
    if not source:
        raise JobValidationError(
            "No adapter or model path. Pick a run's output folder from Runs."
        )
    spec.source_path = str(_check_in_sandbox(source, "Adapter or model path"))
    if spec.export_format not in UI_EXPORT_FORMATS:
        raise JobValidationError(
            f"Unknown export format {spec.export_format!r}; "
            f"one of {', '.join(UI_EXPORT_FORMATS)}."
        )
    if spec.export_format == "gguf" and spec.quantization not in UI_GGUF_QUANTS:
        raise JobValidationError(
            f"Unknown GGUF quantization {spec.quantization!r}; "
            f"one of {', '.join(UI_GGUF_QUANTS)}."
        )
    if spec.ollama_name:
        if spec.export_format != "gguf":
            raise JobValidationError("Ollama registration needs the GGUF format.")
        if not _OLLAMA_NAME_RE.match(spec.ollama_name) or ".." in spec.ollama_name:
            raise JobValidationError(f"Invalid Ollama model name {spec.ollama_name!r}.")


def _validate_spec(spec: JobSpec) -> None:
    """Server-side caps + sandbox checks (handoff rules 6-7). Raises
    JobValidationError with an operator-facing message."""
    _require_types(spec)
    if spec.kind not in ("sft", "multi_run", "export", "calibrate"):
        raise NotImplementedError(f"Unknown job kind {spec.kind!r}.")
    # Before any per-kind return: no UI job runs a model's own code.
    if spec.trust_remote_code:
        raise JobValidationError(
            "trust_remote_code is not available from the UI; run models that "
            "need it from the CLI."
        )
    # Where a job writes is never the client's choice: both overrides must
    # stay inside the sandbox (they exist for tests and server-side callers).
    if spec.output_dir:
        _require_inside_sandbox(spec.output_dir, "Output folder")
    if spec.scratch_root:
        _require_inside_sandbox(spec.scratch_root, "Job folder")
    if spec.kind == "export":
        _validate_export_spec(spec)
        return
    if spec.kind == "calibrate":
        spec.model = _validate_model(spec.model)
        if spec.mode not in ("lora", "full"):
            raise JobValidationError(f"Unknown mode {spec.mode!r}.")
        return
    if spec.kind == "multi_run":
        if not (1 <= int(spec.runs) <= MAX_UI_RUNS):
            raise JobValidationError(f"runs must be 1..{MAX_UI_RUNS} (got {spec.runs}).")
        if spec.merge not in UI_MERGE_CHOICES:
            raise JobValidationError(
                f"Unknown merge {spec.merge!r}; one of {', '.join(UI_MERGE_CHOICES)}."
            )
    if spec.samples is not None and not (1 <= int(spec.samples) <= MAX_UI_SAMPLES):
        raise JobValidationError(
            f"samples must be 1..{MAX_UI_SAMPLES} (got {spec.samples})."
        )
    spec.model = _validate_model(spec.model)
    data = (spec.dataset_path or "").strip()
    if not data:
        raise JobValidationError("No dataset path. Pick one from the Dataset Hub.")
    # The child gets the RESOLVED path: a link swapped after this check
    # cannot point the run somewhere else.
    spec.dataset_path = str(_check_in_sandbox(data, "Dataset"))
    if not (1 <= int(spec.steps) <= MAX_UI_STEPS):
        raise JobValidationError(f"steps must be 1..{MAX_UI_STEPS} (got {spec.steps}).")
    if spec.batch != "auto":
        try:
            batch = int(spec.batch)
        except (TypeError, ValueError):
            raise JobValidationError(f"batch must be 'auto' or an integer (got {spec.batch!r}).") from None
        if not (1 <= batch <= MAX_UI_BATCH):
            raise JobValidationError(f"batch must be 1..{MAX_UI_BATCH} (got {batch}).")
    if not (0.0 < float(spec.lr) <= MAX_UI_LR):
        raise JobValidationError(f"lr must be in (0, {MAX_UI_LR}] (got {spec.lr}).")
    if not (1 <= int(spec.lora_r) <= MAX_UI_LORA_R):
        raise JobValidationError(f"lora_r must be 1..{MAX_UI_LORA_R} (got {spec.lora_r}).")
    if spec.mode not in ("lora", "full"):
        raise JobValidationError(f"Unknown mode {spec.mode!r}.")
    _validate_training_knobs(spec)


def _validate_training_knobs(spec: JobSpec) -> None:
    """Server-side checks for the P3 CLI-parity knobs (handoff rule 7)."""
    if spec.method not in UI_METHODS:
        raise JobValidationError(
            f"Unknown method {spec.method!r}; one of {', '.join(UI_METHODS)}."
        )
    if spec.mode == "full":
        if spec.kind != "sft":
            raise JobValidationError("Full fine-tuning runs as a single run, not a multi-run.")
        if spec.method != "sft":
            raise JobValidationError(
                f"{spec.method.upper()} trains a LoRA adapter; full fine-tuning "
                "supports SFT only."
            )
    allowed = UI_METHOD_PARAMS[spec.method]
    for key, value in dict(spec.method_params or {}).items():
        if key not in allowed:
            raise JobValidationError(f"{key} does not apply to {spec.method.upper()}.")
        _flag, lo, hi = allowed[key]
        if not (lo < float(value) <= hi):
            raise JobValidationError(f"{key} must be in ({lo:g}, {hi:g}] (got {value}).")
    if spec.lora_alpha is not None and not (1 <= int(spec.lora_alpha) <= MAX_UI_LORA_ALPHA):
        raise JobValidationError(
            f"lora_alpha must be 1..{MAX_UI_LORA_ALPHA} (got {spec.lora_alpha})."
        )
    if spec.lora_dropout is not None and not (0.0 <= float(spec.lora_dropout) < 1.0):
        raise JobValidationError(f"lora_dropout must be in [0, 1) (got {spec.lora_dropout}).")
    if spec.target_modules and spec.target_modules != "all-linear":
        parts = spec.target_modules.split(",")
        if len(parts) > 32 or not all(_TARGET_MODULE_RE.match(p) for p in parts):
            raise JobValidationError(
                f"target_modules must be 'all-linear' or up to 32 comma-separated "
                f"module names (got {spec.target_modules!r})."
            )
    if spec.run_name and not _RUN_NAME_RE.match(spec.run_name):
        raise JobValidationError(
            "Run name: up to 128 letters, digits, dots, underscores or dashes."
        )
    if spec.gpu_max_temp is not None:
        lo, hi = UI_GPU_TEMP_RANGE
        if not (lo <= float(spec.gpu_max_temp) <= hi):
            raise JobValidationError(
                f"GPU temperature limit must be {lo:g}..{hi:g} °C (got {spec.gpu_max_temp})."
            )


def vram_verdict(
    model: str,
    *,
    mode: str = "lora",
    lora_r: int = 256,
    batch: str = "auto",
    base_4bit: bool = True,
    gradient_checkpointing: bool = True,
    target_modules: str = "",
    card_gb: float | None = None,
) -> dict[str, Any]:
    """The inline "fits / tight / won't fit" estimate for the training forms.

    The numbers are exactly ``backprop estimate-vram <model> --mode <mode>
    --lora-r <r> --batch-size <b> [--target-modules <t>] [--no-4bit]
    [--no-gradient-checkpointing]``: the same
    :func:`backpropagate.trainer.estimate_vram` call, and for ``batch="auto"``
    the batch the CLI's tier table recommends for this card (what the
    trainer's auto batch picks). Verdict: ``fits`` up to 85 % of the card,
    ``tight`` up to 100 %, ``wont_fit`` above. Never raises: ``unknown``
    with a reason when there is no card or no estimate.
    """
    out: dict[str, Any] = {
        "verdict": "unknown",
        "total_gb": None,
        "card_gb": card_gb,
        "batch": None,
        "note": "",
        "source": "estimate",  # "measured" once this model is calibrated here
    }
    try:
        if not card_gb or card_gb <= 0:
            out["note"] = "No CUDA GPU detected, so there is nothing to compare against."
            return out
        if batch == "auto":
            from .cli import _VRAM_BATCH_SIZE_TIERS, _VRAM_TIER_TOLERANCE_GB

            resolved = _VRAM_BATCH_SIZE_TIERS[-1][1]
            for threshold, bs, _note in _VRAM_BATCH_SIZE_TIERS:
                if card_gb + _VRAM_TIER_TOLERANCE_GB >= threshold:
                    resolved = bs
                    break
        else:
            resolved = int(batch)
        out["batch"] = resolved
        from .trainer import estimate_vram

        kwargs: dict[str, Any] = {"mode": mode, "lora_r": int(lora_r), "batch_size": resolved}
        if not base_4bit and mode == "lora":
            kwargs["quantize_base"] = False
        if not gradient_checkpointing:
            kwargs["gradient_checkpointing"] = False
        if target_modules and mode == "lora":
            kwargs["target_modules"] = target_modules
        estimate = estimate_vram(model, **kwargs)
        total = float(getattr(estimate, "total_gb", 0.0) or 0.0)
        if total <= 0:
            out["note"] = "No estimate for this model."
            return out
        out["total_gb"] = round(total, 2)
        out["source"] = str(getattr(estimate, "source", "estimate"))
        ratio = total / card_gb
        out["verdict"] = "fits" if ratio <= 0.85 else "tight" if ratio <= 1.0 else "wont_fit"
    except Exception as exc:  # noqa: BLE001 — the estimate is advisory
        logger.debug("vram_verdict failed: %r", exc)
        out["verdict"] = "unknown"
        out["note"] = "The estimate is unavailable for this model."
    return out


def _vram_preflight(spec: JobSpec) -> tuple[bool, str]:
    """Best-effort VRAM check. Returns (ok, note). Never raises; on any
    missing signal (no CUDA, CPU-only torch) it returns ok=True with a note
    the UI can surface — the refusal gate needs certainty, not vibes."""
    try:
        import torch

        if not torch.cuda.is_available():
            return True, "no-cuda"
        free_b, _total_b = torch.cuda.mem_get_info()
        # GiB (1024^3): the UI shows the card's capacity in GiB
        # (31.8 for the 32 GB-card 5090), so quoting decimal GB here said
        # "32.5 GB free" on a card that maxes at 31.8.
        free_gib = free_b / (1024**3)
        from .trainer import estimate_vram

        estimate = estimate_vram(
            spec.model,
            mode=spec.mode,
            lora_r=int(spec.lora_r),
            batch_size=1 if spec.batch == "auto" else int(spec.batch),
            quantize_base=bool(spec.base_4bit) or spec.mode != "lora",
            gradient_checkpointing=bool(spec.gradient_checkpointing),
            target_modules=spec.target_modules or None,
        )
        estimate_gb = float(getattr(estimate, "total_gb", 0.0) or 0.0)
        if estimate_gb <= 0:
            return True, "no-estimate"
        if estimate_gb > free_gib:
            return (
                False,
                f"Estimated {estimate_gb:.1f} GB needed but only "
                f"{free_gib:.1f} GB VRAM free. Lower steps/batch or pick a "
                "smaller model.",
            )
        if estimate_gb > 0.9 * free_gib:
            return True, f"tight: ~{estimate_gb:.1f} GB vs {free_gib:.1f} GB free"
        return True, f"fits: ~{estimate_gb:.1f} GB vs {free_gib:.1f} GB free"
    except JobValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 — preflight is advisory
        logger.debug("VRAM preflight skipped: %r", exc)
        return True, "preflight-skipped"


def _posix_process_is_ours(pid: int) -> bool:
    """True when ``pid`` is alive and looks like a backpropagate job.

    POSIX only. ``os.kill(pid, 0)`` probes without signalling; where /proc
    exists the command line must mention backpropagate, so a recycled pid
    that now belongs to something else is not mistaken for the job.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False  # alive, but not ours
    except OSError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True  # no /proc (macOS, BSD): liveness is all we have
    return b"backpropagate" in cmdline


class JobManager:
    """Owns the single active UI job. Process-global (the Reflex app shares
    one instance across tabs).

    Test seams: ``jobs_root``, ``spawn`` (argv -> proc-like with ``pid`` /
    ``poll()`` / ``wait()`` / ``_handle``), ``clock``.
    """

    def __init__(
        self,
        jobs_root: str | Path | None = None,
        spawn: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.jobs_root = Path(jobs_root) if jobs_root else _default_jobs_root()
        # The default root is inside the sandbox and is re-checked at every
        # start; an explicit root is a server-side / test choice.
        self._default_root = jobs_root is None
        self._spawn = spawn or self._default_spawn
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._jobs: dict[str, subprocess.Popen | Any] = {}
        self._handles: dict[str, JobHandle] = {}
        self._win_jobs: dict[str, Any] = {}
        # job id -> when status() first saw its terminal event with the
        # process still alive (see _still_exiting).
        self._exit_wait_since: dict[str, float] = {}

    # ---- lifecycle --------------------------------------------------------

    def start(self, spec: JobSpec) -> JobHandle:
        """Validate, refuse when busy, spawn, and return the handle."""
        _validate_spec(spec)
        # Outside the lock: the estimate can be slow, and Stop must never
        # wait on it. Export loads the model briefly or not at all, and a
        # calibration gates each of its own probes against free VRAM.
        ok, note = (
            (True, spec.kind)
            if spec.kind in ("export", "calibrate")
            else _vram_preflight(spec)
        )
        if not ok:
            raise JobValidationError(note)
        with self._lock:
            active = self._active_handle_locked()
            if active is not None:
                raise JobRefusedError(
                    f"A job is already running ({active.job_id}). "
                    "Stop it or wait for it to finish before starting another."
                )
            orphan = self._running_orphan()
            if orphan is not None:
                raise JobRefusedError(
                    "A job started by an earlier UI session is still running "
                    f"(process {orphan.get('pid')}, {orphan.get('job_id')}). Wait for it "
                    "to finish, or stop that process, before starting another."
                )

            job_id = _mint_job_id(self._clock)
            # Resolve the root right before creating the folder: a symlink or
            # Windows junction planted at <sandbox>/jobs must not redirect
            # the job's files outside the sandbox.
            root = (Path(spec.scratch_root) if spec.scratch_root else self.jobs_root).expanduser()
            root = root.resolve()
            if spec.scratch_root or self._default_root:
                _require_inside_sandbox(str(root), "Job folder")
            run_dir = root / job_id
            run_dir.mkdir(parents=True, exist_ok=False)
            try:
                writer = JobEventWriter(run_dir)
                writer.write({"kind": "phase", "phase": "queued", "note": note})
                log_fh = open(run_dir / OUTPUT_LOG, "ab")  # noqa: SIM115 — owned by proc lifetime
                argv = _build_argv(spec, run_dir)
                # The child gets the server's environment minus its auth
                # secrets (launch token, password verifier): a training
                # process has no use for them, and its log is shown in the UI.
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key not in _CHILD_ENV_DROP
                    and not key.startswith(_CHILD_ENV_DROP_PREFIXES)
                }
                # W&B: #276's report_to=auto now skips an un-configured W&B
                # and honors an opted-in user (wandb login / WANDB_API_KEY);
                # don't force-disable tracking for UI-spawned runs.
                env["PYTHONUNBUFFERED"] = "1"
                proc = self._spawn(argv, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
                handle = JobHandle(
                    job_id=job_id,
                    run_dir=run_dir,
                    pid=int(getattr(proc, "pid", -1)),
                    started_at=self._clock(),
                    spec=spec,
                )
                self._jobs[job_id] = proc
                self._handles[job_id] = handle
                self._write_spawn_record(handle, argv)
                self._assign_crash_lease(handle, proc)
            except Exception:
                # Never leave a half-set-up job occupying the slot.
                self._jobs.pop(job_id, None)
                self._handles.pop(job_id, None)
                raise
        logger.info("UI job started: %s (pid %s)", job_id, handle.pid)
        return handle

    def current(self) -> JobHandle | None:
        with self._lock:
            return self._active_handle_locked()

    def _active_handle_locked(self) -> JobHandle | None:
        # Busy until the process has EXITED, not merely until it wrote its
        # terminal event: a finishing job still holds the GPU while it saves
        # and tears down, and a second job must not start on top of it.
        for job_id, handle in list(self._handles.items()):
            if self._is_alive(job_id):
                return handle
        return None

    def _running_orphan(self) -> dict[str, Any] | None:
        """A job from an earlier UI session whose process is still alive.

        POSIX only: there a UI crash leaves the child running (it has its own
        session), and this manager starts with no memory of it. On Windows the
        Job Object ends the child with the UI server.
        """
        if os.name == "nt":
            return None
        for record in self.scan_orphans():
            job_id = str(record.get("job_id") or "")
            pid = record.get("pid")
            if job_id in self._handles or not isinstance(pid, int) or pid <= 0:
                continue
            if _posix_process_is_ours(pid):
                return record
        return None

    # ---- stop -------------------------------------------------------------

    def stop_current(self, graceful: bool = True, grace_s: float | None = None) -> str:
        """Stop the active job. Returns the resulting status string.

        Contract (captured in tests/test_ui_jobs_manager.py): a job whose
        process already died reports "crashed"; an already-finished job
        reports "done"; a live job gets control.json, then the grace window,
        then a tree kill.
        """
        with self._lock:
            handle = self._active_handle_locked()
            if handle is None:
                # No live job: report the last job's terminal state.
                if self._handles:
                    last = self._handles[list(self._handles)[-1]]
                    return "done" if self._terminal_seen(last) else "crashed"
                return "idle"
            self._request_stop(handle, graceful)
        self._await_exit(handle, self._grace_window(handle, grace_s))
        if self._is_alive(handle.job_id):
            self._hard_kill(handle)
            return "stopped"
        return self._terminal_status(handle)

    # ---- public stop/inspect surface (used by the Reflex layer) ----------

    def get(self, job_id: str) -> JobHandle | None:
        with self._lock:
            return self._handles.get(job_id)

    def is_alive(self, job_id: str) -> bool:
        return self._is_alive(job_id)

    def request_stop(self, job_id: str, graceful: bool = True) -> bool:
        """Write control.json for the job. Returns False when unknown."""
        handle = self.get(job_id)
        if handle is None:
            return False
        self._request_stop(handle, graceful)
        return True

    def grace_window(self, job_id: str) -> float:
        handle = self.get(job_id)
        if handle is None:
            return STOP_GRACE_FLOOR_S
        return self._grace_window(handle, None)

    def cancel(self, job_id: str) -> None:
        """Kill a job that has no cooperative stop (export) and record it as
        stopped, so the page reads "cancelled", not "crashed"."""
        handle = self.get(job_id)
        self.hard_kill(job_id)
        if handle is not None and not self._terminal_seen(handle):
            JobEventWriter(handle.run_dir).done(status="stopped", steps_done=0)

    def hard_kill(self, job_id: str) -> None:
        handle = self.get(job_id)
        if handle is not None:
            self._hard_kill(handle)

    def _request_stop(self, handle: JobHandle, graceful: bool) -> None:
        if graceful:
            payload = {"action": "stop_save", "ts": self._clock()}
            try:
                (handle.run_dir / CONTROL_FILENAME).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            except OSError as exc:
                logger.warning("control.json write failed for %s: %s", handle.job_id, exc)

    def _grace_window(self, handle: JobHandle, override: float | None) -> float:
        if override is not None:
            return float(override)
        last_step_ms = self._last_step_time_ms(handle)
        if last_step_ms and last_step_ms > 0:
            return max(STOP_GRACE_FLOOR_S, 3.0 * last_step_ms / 1000.0 + 20.0)
        return STOP_GRACE_FLOOR_S

    def _await_exit(self, handle: JobHandle, grace_s: float) -> None:
        # Wall clock, not the injected clock: the injected clock is for
        # timestamps; sleeping against it would make a fake clock crawl.
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and self._is_alive(handle.job_id):
            time.sleep(0.25)

    def _hard_kill(self, handle: JobHandle) -> None:
        proc = self._jobs.get(handle.job_id)
        if proc is None:
            return
        try:
            from .cli import _kill_process_tree

            if isinstance(proc, subprocess.Popen):
                _kill_process_tree(proc)
            else:  # test doubles / foreign proc-likes
                proc.kill()
        except Exception:  # noqa: BLE001
            try:
                if hasattr(proc, "kill"):
                    proc.kill()
            except Exception:  # noqa: BLE001  # nosec B110 — last resort
                pass
        logger.warning("UI job %s escalated to tree kill", handle.job_id)

    # ---- status / observability ---------------------------------------------

    def _is_alive(self, job_id: str) -> bool:
        proc = self._jobs.get(job_id)
        if proc is None:
            return False
        try:
            return proc.poll() is None
        except Exception:  # noqa: BLE001 — fake procs in tests
            return bool(getattr(proc, "_alive", False))

    def _tail_rows(self, handle: JobHandle) -> list[dict]:
        """The last rows of events.jsonl (a fixed window). The terminal event
        is the last thing a job writes, so this is enough to find it without
        reading a long run's whole file."""
        try:
            size = handle.events_path.stat().st_size
        except OSError:
            return []
        # Starting mid-file cuts the first line; it fails to parse and is skipped.
        rows, _ = self.tail_events(handle, max(0, size - _EVENT_TAIL_BYTES))
        return rows

    def _iter_events(self, handle: JobHandle) -> Iterator[dict]:
        """Every row of events.jsonl, read a bounded slice at a time."""
        offset = 0
        while True:
            rows, new_offset = self.tail_events(handle, offset)
            yield from rows
            if new_offset <= offset:
                return
            offset = new_offset

    def _terminal_seen(self, handle: JobHandle) -> bool:
        return any(row.get("kind") in ("done", "error") for row in self._tail_rows(handle))

    def _terminal_status(self, handle: JobHandle) -> str:
        status = "crashed"
        for row in self._tail_rows(handle):
            if row.get("kind") == "done":
                status = str(row.get("status") or "done")
            elif row.get("kind") == "error":
                status = "failed"
        if not self._terminal_seen(handle) and not self._is_alive(handle.job_id):
            return "crashed"
        return status

    def _still_exiting(self, job_id: str) -> bool:
        """True while a job that wrote its terminal event may still be
        tearing down (saving, releasing the GPU). Bounded: after
        ``EXIT_GRACE_S`` the job is reported by its terminal event anyway, so
        a process that never exits cannot hold the page in "active"."""
        now = time.monotonic()
        first = self._exit_wait_since.setdefault(job_id, now)
        return now - first < EXIT_GRACE_S

    def status(self) -> dict[str, Any]:  # noqa: C901 — small presentation gather
        """Latest status dict for the side rail / progress banner."""
        with self._lock:
            handle = self._handles[list(self._handles)[-1]] if self._handles else None
        if handle is None:
            return {"status": "idle"}
        latest: dict[str, Any] = {}
        phase = "queued"
        terminal = False
        for row in self._iter_events(handle):  # streamed: constant memory
            kind = row.get("kind")
            if kind == "phase":
                phase = str(row.get("phase", phase))
            elif kind == "step" or kind in ("done", "error"):
                latest = row
                terminal = terminal or kind in ("done", "error")
        alive = self._is_alive(handle.job_id)
        finishing = alive and terminal and self._still_exiting(handle.job_id)
        if not alive and not terminal:
            state = "crashed"
        elif finishing:
            # Reported finished only once the process is gone: start()
            # refuses a new job until then, so saying "done" earlier offered
            # Start for a moment and then refused the click.
            state = "active"
        elif latest.get("kind") == "done":
            state = str(latest.get("status") or "done")
        elif latest.get("kind") == "error":
            state = "failed"
        elif alive:
            state = "active"
        else:
            state = "crashed"
        last_step_ms = latest.get("step_time_ms")
        out: dict[str, Any] = {
            "job_id": handle.job_id,
            "kind": getattr(handle.spec, "kind", "sft"),
            "status": state,
            "phase": phase,
            "step": int(latest.get("step") or latest.get("steps_done") or 0),
            "total_steps": int(latest.get("total_steps") or 0),
            "loss": latest.get("loss"),
            "ema_loss": latest.get("ema_loss"),
            "lr": latest.get("lr"),
            "it_s": (1000.0 / last_step_ms) if last_step_ms else None,
            "vram_alloc_gib": latest.get("vram_alloc_gib"),
            "vram_reserved_gib": latest.get("vram_reserved_gib"),
            "temp_c": latest.get("temp_c"),
            "started_at": handle.started_at,
            "log_path": str(handle.log_path),
            "events_path": str(handle.events_path),
            "finishing": finishing,
        }
        for key in ("code", "message", "hint", "output_path"):
            if latest.get(key) is not None:
                out[key] = latest.get(key)
        return out

    def tail_events(self, handle_or_id: JobHandle | str, offset: int = 0) -> tuple[list[dict], int]:
        """Read new events.jsonl rows from byte ``offset``. Returns
        (rows, new_offset); tolerates a partially-written trailing line."""
        handle = (
            handle_or_id
            if isinstance(handle_or_id, JobHandle)
            else self._handles.get(str(handle_or_id))
        )
        if handle is None:
            return [], offset
        path = handle.events_path
        try:
            with open(path, "rb") as fh:
                fh.seek(offset)
                # Bounded: a very long (or hostile) file is read a slice per
                # poll; the next poll continues from the returned offset.
                chunk = fh.read(MAX_EVENT_READ_BYTES)
        except FileNotFoundError:
            return [], offset
        except OSError:
            return [], offset
        if not chunk:
            return [], offset
        lines = chunk.split(b"\n")
        tail_complete = chunk.endswith(b"\n")
        if not tail_complete:
            lines = lines[:-1]  # hold the partial line for next poll
        rows: list[dict] = []
        consumed = 0
        for raw in lines:
            consumed += len(raw) + 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                logger.debug("ui job event parse skipped: %r", raw[:120])
                continue
            # Every consumer treats a row as a mapping; anything else in the
            # file (a list, a bare string) is not an event.
            if isinstance(row, dict):
                rows.append(row)
        new_offset = offset + (len(chunk) if tail_complete else consumed)
        return rows, new_offset

    def _read_events(self, handle: JobHandle, limit: int | None = 200) -> list[dict]:
        """The last ``limit`` rows (from the tail window), or every row when
        ``limit`` is None (streamed; avoid on hot paths)."""
        if limit is None:
            return list(self._iter_events(handle))
        rows = self._tail_rows(handle)
        return rows[-limit:] if len(rows) > limit else rows

    def _last_step_time_ms(self, handle: JobHandle) -> float | None:
        for row in reversed(self._read_events(handle, limit=50)):
            if row.get("kind") == "step" and row.get("step_time_ms"):
                try:
                    return float(row["step_time_ms"])
                except (TypeError, ValueError):
                    return None
        return None

    # ---- spawn plumbing -------------------------------------------------------

    def _default_spawn(
        self, argv: list[str], stdout: Any, stderr: Any, env: dict[str, str]
    ) -> subprocess.Popen[Any]:  # noqa: ARG002
        kwargs: dict[str, Any] = {"stdout": stdout, "stderr": stderr, "env": env}
        if os.name == "nt":
            # Own process group so tree-kill targets it; no console window of
            # its own (child of the console the UI runs in).
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(argv, **kwargs)

    def _write_spawn_record(self, handle: JobHandle, argv: list[str]) -> None:
        record = {
            "job_id": handle.job_id,
            "kind": handle.spec.kind,
            "pid": handle.pid,
            "started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(handle.started_at)
            ),
            "started_at_epoch": handle.started_at,
            "status": "running",
            "argv_tail": _redact_argv([str(a) for a in argv[4:]]),
        }
        try:
            (handle.run_dir / JOB_FILENAME).write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("job.json spawn record failed for %s: %s", handle.job_id, exc)

    def _assign_crash_lease(self, handle: JobHandle, proc: Any) -> None:
        """Windows: place the child in a kill-on-close Job Object so a UI
        crash takes the whole training tree down with it."""
        if os.name != "nt":
            return
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            class _BasicLimit(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32),
                ]

            class _IoCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
                )]

            class _ExtendedLimit(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BasicLimit),
                    ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
            JobObjectExtendedLimitInformation = 9
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                logger.warning("CreateJobObjectW failed for %s", handle.job_id)
                return
            info = _ExtendedLimit()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            proc_handle = int(getattr(proc, "_handle", 0) or 0)
            if ok and proc_handle:
                ok = kernel32.AssignProcessToJobObject(job, proc_handle)
            if not ok:
                logger.warning(
                    "Job Object assignment failed for %s — teardown falls "
                    "back to taskkill /T.", handle.job_id
                )
                kernel32.CloseHandle(job)
                return
            self._win_jobs[handle.job_id] = job
        except Exception as exc:  # noqa: BLE001 — platform variance must not block start
            logger.warning("crash-lease setup failed for %s: %r", handle.job_id, exc)

    def close(self) -> None:
        """Shut down: closing the Job Object handle reaps live children on
        Windows (kill-on-close); POSIX children get the tree kill."""
        for job_id, job in list(self._win_jobs.items()):
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(job)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001  # nosec B110
                pass
            self._win_jobs.pop(job_id, None)
        for _job_id, proc in list(self._jobs.items()):
            try:
                if proc.poll() is None:
                    from .cli import _kill_process_tree

                    _kill_process_tree(proc)
            except Exception:  # noqa: BLE001  # nosec B110 — shutdown path
                pass
        self._jobs.clear()

    # ---- orphan scan (UI restart while a child outlives us) ----------------

    def _storage_root(self) -> Path | None:
        """The resolved jobs folder, or None when it does not exist or (for the
        default root) is not inside the sandbox."""
        root = self.jobs_root.expanduser()
        if not root.exists():
            return None
        root = root.resolve()
        if self._default_root:
            try:
                _require_inside_sandbox(str(root), "Job folder")
            except JobValidationError:
                return None
        return root

    def _removable(self, folder: Path) -> bool:
        """True for a job folder the clean-up may remove: no saved model, not
        the job that is running, not a job an earlier UI session left running."""
        with self._lock:
            active = self._active_handle_locked()
        if active is not None and folder.name == active.job_id:
            return False
        if _has_saved_model(folder):
            return False
        if os.name != "nt":
            try:
                record = json.loads((folder / JOB_FILENAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            pid = record.get("pid") if isinstance(record, dict) else None
            if (
                isinstance(record, dict)
                and record.get("status") == "running"
                and isinstance(pid, int)
                and pid > 0
                and _posix_process_is_ours(pid)
            ):
                return False
        return True

    def storage_summary(self) -> dict[str, Any]:
        """What the job folders use on disk, and how much the clean-up would
        free. Nothing is ever removed automatically."""
        out: dict[str, Any] = {
            "folders": 0, "bytes": 0, "removable_folders": 0, "removable_bytes": 0,
        }
        root = self._storage_root()
        if root is None:
            return out
        for folder in _job_folders(root):
            size = _folder_bytes(folder)
            out["folders"] += 1
            out["bytes"] += size
            if self._removable(folder):
                out["removable_folders"] += 1
                out["removable_bytes"] += size
        return out

    def clean_up(self) -> dict[str, Any]:
        """Remove job folders that hold no saved model (failed, cancelled and
        measurement jobs). A folder with a model is never touched here; neither
        is a running job, a link, or anything outside the jobs folder."""
        import shutil

        out: dict[str, Any] = {"removed": 0, "bytes": 0, "errors": 0}
        root = self._storage_root()
        if root is None:
            return out
        for folder in _job_folders(root):
            if folder.parent != root or not self._removable(folder):
                continue
            size = _folder_bytes(folder)
            try:
                shutil.rmtree(folder)
            except OSError:
                out["errors"] += 1
                continue
            self._handles.pop(folder.name, None)
            self._jobs.pop(folder.name, None)
            out["removed"] += 1
            out["bytes"] += size
        return out

    def scan_orphans(self) -> list[dict[str, Any]]:
        """List run dirs whose job.json claims a running job. The caller
        checks pid liveness (psutil where available) and marks dead pids
        'crashed'. P1: read-only reporting — no kill here."""
        out: list[dict[str, Any]] = []
        root = self.jobs_root
        if not root.exists():
            return out
        for child in sorted(root.iterdir()):
            job_file = child / JOB_FILENAME
            if not job_file.exists():
                continue
            try:
                record = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("status") == "running":
                out.append(record)
        return out


# ---- storage: what the job folders use, and a clean-up that never removes a model ----

#: A job folder that holds any of these has a saved model and is never
#: removed by the clean-up (delete such a run from its own page).
_MODEL_FILE_SUFFIXES = (".safetensors", ".gguf")
_MODEL_FILE_NAMES = ("adapter_config.json", "adapter_model.bin", "pytorch_model.bin")


def _is_link(path: Path) -> bool:
    """A symlink or (Windows) junction. Never followed, never removed."""
    try:
        if path.is_symlink():
            return True
        isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
        if isjunction is not None and isjunction(path):
            return True
        # Older Pythons: a junction resolves somewhere other than its own path.
        return path.resolve() != path.parent.resolve() / path.name
    except OSError:
        return True


def _own_files(folder: Path) -> Iterator[Path]:
    """Every file under ``folder``, without entering a symlink or junction.

    Not ``os.walk``: with ``followlinks=False`` it still descends into Windows
    junctions, which would count (and judge the folder by) files that live
    somewhere else.
    """
    stack = [folder]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if _is_link(child):
                continue
            try:
                if child.is_dir():
                    stack.append(child)
                else:
                    yield child
            except OSError:
                continue


def _folder_bytes(folder: Path) -> int:
    total = 0
    for path in _own_files(folder):
        try:
            total += path.lstat().st_size
        except OSError:
            continue
    return total


def _has_saved_model(folder: Path) -> bool:
    for path in _own_files(folder):
        lower = path.name.lower()
        if lower.endswith(_MODEL_FILE_SUFFIXES) or lower in _MODEL_FILE_NAMES:
            return True
    return False


def _job_folders(root: Path) -> list[Path]:
    """Real ``run_*`` directories directly under ``root`` (links skipped)."""
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    return [
        child
        for child in children
        if child.name.startswith("run_") and not _is_link(child) and child.is_dir()
    ]


_MANAGER_SINGLETON: JobManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_job_manager() -> JobManager:
    """Process-global manager shared by every Reflex state instance."""
    global _MANAGER_SINGLETON
    with _MANAGER_LOCK:
        if _MANAGER_SINGLETON is None:
            _MANAGER_SINGLETON = JobManager()
        return _MANAGER_SINGLETON


__all__ = [
    "JOB_FILENAME",
    "MAX_UI_BATCH",
    "MAX_UI_LORA_R",
    "MAX_UI_LR",
    "MAX_UI_RUNS",
    "MAX_UI_SAMPLES",
    "MAX_UI_SEQ_LENGTH",
    "MAX_UI_STEPS",
    "OUTPUT_LOG",
    "STOP_GRACE_FLOOR_S",
    "UI_EXPORT_FORMATS",
    "UI_GGUF_QUANTS",
    "UI_MERGE_CHOICES",
    "JobHandle",
    "JobManager",
    "JobRefusedError",
    "JobSpec",
    "JobValidationError",
    "get_job_manager",
]
