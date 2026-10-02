"""Measure training VRAM on the GPU that is running, and reuse the result.

``backprop estimate-vram <model> --calibrate`` (and "Measure on this GPU" in
the web UI) run a few very short REAL training probes of one model and fit
that model's memory cost on this machine:

    peak(batch, seq) = load + max(floor, batch * seq * bytes_per_token)

``floor`` is a fixed cost every run pays (a temporary full-precision copy of
the embedding table: 0.98 GiB for Llama 3.2 1B, 2.03 GiB for Qwen2.5 7B), so
small probes all read the same and say nothing about the per-row cost. The
probes are therefore chosen ABOVE the floor.

The formula shipped in :func:`backpropagate.trainer.estimate_vram` has the
same shape with coefficients fitted on one RTX 5090. Another card, driver,
torch build or attention kernel shifts them; a calibration replaces them with
this machine's own numbers for this model.

Safety rules (a probe must never take the machine down):

* The child caps its own GPU memory
  (``torch.cuda.set_per_process_memory_fraction``) below what was free at
  start, less a headroom of 1.5 GiB or 8% of the card, whichever is larger.
  On Windows an uncapped overrun does not fail: the driver spills into
  shared system memory and the whole desktop stutters. Capped, it raises a
  clean out-of-memory error instead. With less than 1.5 GiB left after the
  headroom nothing is run.
* The cap covers PyTorch's own allocator only. A paged optimizer keeps its
  state in CUDA managed memory outside it (and on Windows that memory can
  stall the desktop), so the probes always train with the non-paged
  ``adamw_8bit``.
* The child dies with its parent: a kill-on-close Job Object on Windows, its
  own session on POSIX, killed when the parent stops waiting.
* Each probe runs only if it is PREDICTED to fit in the budget: by the shipped
  formula on the loaded model's real shape, scaled up by what the probes
  already measured. Nothing tests the limit.
* When no informative probe fits (a big model on a small card), the load size
  and the floor are still measured and stored; the per-row cost then stays the
  formula's.
* An out-of-memory probe is recorded and ends the run; what was measured up
  to there is still used.

Stored per GPU + model + mode + base precision + library versions in
``~/.backpropagate/vram-calibration.json`` (``BACKPROPAGATE_VRAM_CALIBRATION``
overrides the path), so a new card, driver stack or model starts clean.

This module imports nothing heavy at import time; torch and the trainer load
only inside the functions that need them.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 3: records the optimizer, the probed row lengths and, for full
#: fine-tuning, the fixed training cost. 2: the per-row cost is the highest
#: per-token cost over the probes. 1: a two-term least-squares fit made
#: before the efficient-attention fix.
CALIBRATION_VERSION = 3
GIB = 1024**3

#: Candidate (batch, seq) probes, most informative first. Up to
#: ``MAX_PROBES`` run: those predicted to clear the floor and fit the budget.
PROBE_CANDIDATES: tuple[tuple[int, int], ...] = (
    (2, 2048), (4, 2048), (4, 1024), (1, 2048), (8, 1024), (2, 1024),
)
MAX_PROBES = 3
#: A probe is informative when its predicted row cost clears the floor by this.
FLOOR_CLEARANCE = 1.3
#: A probe runs only if its predicted peak is within this share of the budget.
PROBE_BUDGET_SHARE = 0.8
#: GPU memory left untouched below what was free at start: this many GiB or
#: this share of the card, whichever is larger.
HEADROOM_MIN_GIB = 1.5
HEADROOM_SHARE = 0.08
#: Below this budget (free minus headroom) nothing is run.
MIN_BUDGET_GIB = 1.5
#: The cap is never set above this share of the card.
MAX_FRACTION = 0.95
#: The probes' optimizer. Never a paged one: see the module docstring.
PROBE_OPTIM = "adamw_8bit"
#: Full fine-tuning: bytes per parameter held beyond the weights (gradients
#: + 8-bit optimizer state; measured 5.45 on Llama 3.2 1B, 2026-10-02). Used
#: to decide whether a probe is predicted to fit.
FULL_FT_TRAIN_BYTES = 5.5
#: A measurement is used for rows up to this many times the longest row it
#: probed (2,048-token probes predicted a 3,241-token run within 0.3%).
SEQ_RANGE_FACTOR = 2.0
#: The probes train a tiny adapter so the save is small; the user's own
#: adapter size is added analytically (it is an exact count of parameters).
PROBE_LORA_R = 8
PROBE_TARGETS = ("q_proj", "v_proj")
PROBE_STEPS = 2
#: Allocator slack over the peak allocation (measured ~6% on real runs).
MEASURED_OVERHEAD = 0.06
#: Bytes per trainable adapter parameter: fp32 weights, then gradients +
#: optimizer state while training (measured, see estimate_vram).
ADAPTER_BYTES_LOADED = 4.0
ADAPTER_BYTES_TRAINING = 8.4


class CalibrationError(RuntimeError):
    """Calibration could not run or could not be fitted. The message is
    operator-safe."""


@dataclass
class Calibration:
    """One model's measured memory cost on one machine."""

    model: str
    mode: str  # "lora" | "full"
    base_4bit: bool
    machine: dict[str, Any]
    load_gib: float  # weights (+ the tiny probe adapter) after loading
    floor_gib: float  # fixed transient every run pays (fp32 embedding copy)
    # Per row: bytes per token (``quad_bytes``, per token^2, is always 0.0 and
    # kept for the stored format). None when no probe above the floor fitted
    # this card; the estimate then uses the formula's row cost.
    quad_bytes: float | None
    lin_bytes: float | None
    probe_trainable_params: int
    probes: list[dict[str, Any]] = field(default_factory=list)
    max_residual_pct: float = 0.0
    measured_at: str = ""
    version: int = CALIBRATION_VERSION
    # Full fine-tuning: gradients + optimizer state, the same for every batch.
    # 0.0 for LoRA (the adapter's own cost is added from its parameter count).
    fixed_gib: float = 0.0
    optim: str = ""  # the optimizer the probes trained with
    seq_min: int = 0  # shortest and longest row length probed (0: none)
    seq_max: int = 0

    @property
    def rows_measured(self) -> bool:
        return self.quad_bytes is not None and self.lin_bytes is not None

    def covers(self, seq: int) -> bool:
        """Is a row of ``seq`` tokens within reach of what was probed?"""
        return self.seq_max <= 0 or seq <= self.seq_max * SEQ_RANGE_FACTOR

    def rows_gib(self, batch: int, seq: int) -> float | None:
        """Measured cost of ``batch`` rows of ``seq`` tokens (None if not measured)."""
        if self.quad_bytes is None or self.lin_bytes is None:
            return None
        return batch * (self.quad_bytes * seq * seq + self.lin_bytes * seq) / GIB


# ---- storage --------------------------------------------------------------------


def calibration_path() -> Path:
    override = os.environ.get("BACKPROPAGATE_VRAM_CALIBRATION", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".backpropagate" / "vram-calibration.json"


def _lib_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - not installed / odd metadata
        return ""


def machine_fingerprint() -> dict[str, Any] | None:
    """What a calibration is valid for: the GPU and the libraries that decide
    how memory is used. None without a CUDA GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / GIB
        name = torch.cuda.get_device_name(device)
        torch_version = str(torch.__version__)
    except Exception:  # noqa: BLE001 - no torch / broken CUDA
        return None
    from .trainer import _varlen_attention_installed

    return {
        "gpu": name,
        "vram_gib": round(total, 1),
        "torch": torch_version,
        "transformers": _lib_version("transformers"),
        "unsloth": _lib_version("unsloth"),
        # Which attention kernel family trains here. Measurements taken with
        # one do not describe the other.
        "attention": "varlen" if _varlen_attention_installed() else "sdpa-efficient",
    }


def _key(model: str, mode: str, base_4bit: bool, machine: dict[str, Any]) -> str:
    return json.dumps(
        {
            "model": model.strip().lower(),
            "mode": mode,
            "base_4bit": bool(base_4bit) if mode == "lora" else False,
            "machine": machine,
        },
        sort_keys=True,
    )


class _StoreUnreadable(Exception):
    """The store exists but is not a JSON object."""


def _read_store(strict: bool = False) -> dict[str, Any]:
    """The stored measurements. A missing file is an empty store. A file that
    exists but cannot be parsed is an empty store for readers, and raises
    ``_StoreUnreadable`` for the writer (``strict``), which must not replace
    it blindly."""
    path = calibration_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        if strict:
            raise _StoreUnreadable(str(exc)) from exc
        return {}
    try:
        data = json.loads(text)
    except ValueError as exc:
        if strict:
            raise _StoreUnreadable(str(exc)) from exc
        return {}
    if not isinstance(data, dict):
        if strict:
            raise _StoreUnreadable("not a JSON object")
        return {}
    return data


def _usable(cal: Calibration) -> bool:
    """A stored row with numbers an estimate can be built from."""
    import math

    def number(value: Any, minimum: float) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= minimum
        )

    if not number(cal.load_gib, 0.01) or not number(cal.floor_gib, 0.0):
        return False
    if not number(cal.fixed_gib, 0.0) or not number(cal.probe_trainable_params, 0):
        return False
    if cal.lin_bytes is not None and not number(cal.lin_bytes, 1e-9):
        return False
    if cal.quad_bytes is not None and not number(cal.quad_bytes, 0.0):
        return False
    return number(cal.seq_min, 0) and number(cal.seq_max, 0)


def lookup(
    model: str, mode: str = "lora", base_4bit: bool = True, machine: dict[str, Any] | None = None
) -> Calibration | None:
    """The stored calibration for this model on THIS machine, or None."""
    store = _read_store()
    if not store:
        return None
    machine = machine if machine is not None else machine_fingerprint()
    if machine is None:
        return None
    raw = store.get(_key(model, mode, base_4bit, machine))
    if not isinstance(raw, dict) or raw.get("version") != CALIBRATION_VERSION:
        return None
    try:
        cal = Calibration(**raw)
    except TypeError:
        return None
    return cal if _usable(cal) else None


#: How long save() waits for another writer, and when a lock file left behind
#: by a dead writer is taken over.
_LOCK_WAIT_S = 10.0
_LOCK_STALE_S = 60.0


def _acquire_lock(lock: Path) -> bool:
    deadline = time.monotonic() + _LOCK_WAIT_S
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE_S:
                    lock.unlink()
                    continue
            except OSError:
                continue
        except OSError:
            return False
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def save(cal: Calibration) -> Path:
    """Add one measurement to the store.

    One writer at a time (a lock file), a temporary file of its own, and an
    atomic replace. A store that exists but cannot be read is moved aside to
    ``<name>.unreadable`` rather than overwritten.
    """
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    locked = _acquire_lock(lock)
    try:
        try:
            store = _read_store(strict=True)
        except _StoreUnreadable as exc:
            aside = path.with_name(path.name + ".unreadable")
            logger.warning(
                "The VRAM calibration store could not be read (%s); it was moved to %s "
                "and a new one started.", exc, aside.name,
            )
            try:
                os.replace(path, aside)
            except OSError as move_exc:
                raise CalibrationError(
                    f"The calibration store at {path} cannot be read or moved aside "
                    f"({move_exc}). Delete it, or point BACKPROPAGATE_VRAM_CALIBRATION "
                    "somewhere else."
                ) from move_exc
            store = {}
        store[_key(cal.model, cal.mode, cal.base_4bit, cal.machine)] = asdict(cal)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        try:
            tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    finally:
        if locked:
            try:
                lock.unlink()
            except OSError:
                pass
    return path


# ---- the fit -----------------------------------------------------------------------


def fit_rows(
    probes: list[dict[str, Any]], formula_quad: float = 0.0, formula_lin: float = 0.0  # noqa: ARG001
) -> tuple[float, float, float] | None:
    """Per-row ``(quad_bytes, lin_bytes, spread_pct)`` from the probes.

    ``probes`` rows carry ``batch``, ``seq`` and ``overhead_gib`` (peak minus
    the loaded baseline), all above the floor. Memory is linear in the tokens
    of a batch, so each probe gives one cost per token; the HIGHEST is kept
    (``quad_bytes`` is always 0.0 and stays only for the stored format).

    Why the highest and not a least-squares fit: Unsloth compiles its loss,
    and the compiled version needs about half the logits memory. The first
    training call in a process can run before that takes effect, which is
    what a real run's first steps do too, so later probes in the same process
    read lower than a fresh run peaks (Llama 3.2 1B: 555 KB per token on the
    first probe, about 300 KB on the next two). An estimate must cover the
    fresh run. ``spread_pct`` is how far the lowest probe sits below the
    highest. None when nothing usable was measured.

    ``formula_quad`` / ``formula_lin`` are accepted for older callers and
    ignored.
    """
    per_token = [
        float(p["overhead_gib"]) * GIB / (float(p["batch"]) * float(p["seq"]))
        for p in probes
        if not p.get("oom") and p.get("overhead_gib") and p.get("batch") and p.get("seq")
    ]
    per_token = [value for value in per_token if value > 0]
    if not per_token:
        return None
    highest, lowest = max(per_token), min(per_token)
    return 0.0, highest, (highest - lowest) / highest * 100.0


def fit_fixed(probes: list[dict[str, Any]], bytes_per_token: float) -> float:
    """Full fine-tuning: the cost that does not depend on the batch (GiB).

    A full fine-tune's overhead is gradients plus optimizer state, the same
    for every batch, plus the batch's rows. The rows are priced at the
    formula's ``bytes_per_token`` and the largest remainder over the probes
    is the fixed part, so no probe is under-predicted.
    """
    remainders = [
        float(p["overhead_gib"]) - float(p["batch"]) * float(p["seq"]) * bytes_per_token / GIB
        for p in probes
        if not p.get("oom") and p.get("overhead_gib") and p.get("batch") and p.get("seq")
    ]
    return max(0.0, max(remainders)) if remainders else 0.0


def probe_budget(free_gib: float, total_gib: float) -> tuple[float, float] | None:
    """``(budget_gib, fraction)`` for the probe process, or None when too
    little GPU memory is free to run anything.

    The fraction is of the whole card, as PyTorch wants it, and is never set
    so that ``fraction * total`` exceeds what is free minus the headroom.
    """
    if total_gib <= 0:
        return None
    headroom = max(HEADROOM_MIN_GIB, HEADROOM_SHARE * total_gib)
    budget = free_gib - headroom
    if budget < MIN_BUDGET_GIB:
        return None
    return budget, min(MAX_FRACTION, budget / total_gib)


# ---- running it (parent side) ----------------------------------------------------------


def calibrate(
    model: str,
    *,
    mode: str = "lora",
    base_4bit: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    timeout_s: float = 3600.0,
) -> Calibration:
    """Run the probes in a child process, fit, store and return the result.

    ``on_event`` receives ``{"event": "loading" | "loaded" | "probe" |
    "skipped", ...}`` rows as the child reports them.
    """
    machine = machine_fingerprint()
    if machine is None:
        raise CalibrationError("No CUDA GPU detected: there is nothing to measure on.")
    payload = {"model": model, "mode": mode, "base_4bit": bool(base_4bit)}
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    env["PYTHONIOENCODING"] = "utf-8"
    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True  # its own group: killable as a tree
    proc = subprocess.Popen(  # noqa: S603 - argv built here, no shell
        [sys.executable, "-m", "backpropagate.vram_calibration", json.dumps(payload)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        **popen_kwargs,
    )
    # The probe must not outlive this process: a Job Object that kills it
    # when our handle closes (Windows), a tree kill on the way out (POSIX).
    lease = _kill_on_close(proc)
    result: dict[str, Any] | None = None
    tail: list[str] = []
    # A timer, not a check in the read loop: a child that hangs silently
    # produces no line to wake the loop, and must still be stopped.
    timed_out = threading.Event()

    def _expire() -> None:
        timed_out.set()
        _kill_tree(proc)

    timer = threading.Timer(timeout_s, _expire)
    timer.daemon = True
    timer.start()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            # Bytes from the pipe, decoded here: the trainer's progress bars are
            # not always valid UTF-8 on Windows consoles.
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            line = line.rstrip("\r\n")
            if line.startswith("CALIBRATION "):
                try:
                    row = json.loads(line[len("CALIBRATION "):])
                except ValueError:
                    continue
                if row.get("event") == "result":
                    result = row
                elif on_event is not None:
                    on_event(row)
            else:
                tail = [*tail[-39:], line]
        rc = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:  # interrupted, or on_event raised
            _kill_tree(proc)
        _release(lease)
    # A result that arrived is kept even if the timer fired as it did.
    if result is None and timed_out.is_set():
        raise CalibrationError(
            f"Calibration did not finish within {timeout_s / 60:.0f} minutes and was stopped."
        )
    if result is None:
        detail = " | ".join(t for t in tail[-6:] if t.strip())[-600:]
        raise CalibrationError(
            f"The calibration process ended without a result (exit {rc}). {detail}"
        )
    if result.get("error"):
        raise CalibrationError(str(result["error"]))
    fitted = fit_rows(
        result["probes"], float(result["formula_quad"]), float(result["formula_lin"])
    )
    quad, lin, resid = fitted if fitted is not None else (None, None, 0.0)
    good = [p for p in result["probes"] if not p.get("oom") and p.get("overhead_gib")]
    fixed = 0.0
    if mode == "full" and good:
        # Rows at the formula's cost per token; what is left is the fixed
        # part (gradients + optimizer state).
        quad, lin, resid = 0.0, float(result["formula_lin"]), 0.0
        fixed = fit_fixed(good, lin)
    cal = Calibration(
        model=model,
        mode=mode,
        base_4bit=bool(base_4bit) if mode == "lora" else False,
        machine=machine,
        load_gib=float(result["load_gib"]),
        floor_gib=float(result["floor_gib"]),
        quad_bytes=quad,
        lin_bytes=lin,
        probe_trainable_params=int(result.get("trainable_params") or 0),
        probes=list(result["probes"]),
        max_residual_pct=round(resid, 2),
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        fixed_gib=round(fixed, 4),
        optim=str(result.get("optim") or ""),
        seq_min=min((int(p["seq"]) for p in good), default=0),
        seq_max=max((int(p["seq"]) for p in good), default=0),
    )
    save(cal)
    return cal


def _kill_on_close(proc: Any) -> Any:
    """Windows: put ``proc`` in a Job Object that kills it when the handle
    closes, which the OS does when this process ends for any reason. Returns
    the job handle (keep it alive) or None."""
    if os.name != "nt":
        return None
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

        handle = int(getattr(proc, "_handle", 0) or 0)
        if not handle:
            return None
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info)
        )
        if ok:
            ok = kernel32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(handle))
        if not ok:
            kernel32.CloseHandle(ctypes.c_void_p(job))
            return None
        return job
    except Exception as exc:  # noqa: BLE001 - a missing lease must not stop a calibration
        logger.debug("kill-on-close lease failed: %r", exc)
        return None


def _release(lease: Any) -> None:
    if lease is None or os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(lease))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001  # nosec B110
        pass


def _kill_tree(proc: Any) -> None:
    """Stop the probe process and anything it started."""
    pid = getattr(proc, "pid", None)
    try:
        if not isinstance(pid, int) or pid <= 0:
            pass
        elif os.name == "nt":
            subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(pid), "/T", "/F"],  # noqa: S607
                capture_output=True, timeout=30, check=False,
            )
        else:
            import signal

            os.killpg(  # type: ignore[attr-defined,unused-ignore]
                os.getpgid(pid),  # type: ignore[attr-defined,unused-ignore]
                getattr(signal, "SIGKILL", signal.SIGTERM),
            )
    except Exception:  # noqa: BLE001 - fall back to the one process
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001  # nosec B110
        pass


# ---- the child ------------------------------------------------------------------------


def _emit(row: dict[str, Any]) -> None:
    print("CALIBRATION " + json.dumps(row), flush=True)


def _probe_dataset(directory: Path, seq: int, rows: int) -> Path:
    """Chat rows longer than ``seq`` tokens, so every row truncates to ``seq``."""
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "and", "keeps", "running", "across", "the", "long", "field", "while", "the", "sun", "sets", "slowly", "behind", "distant", "hills"]
    out = directory / f"probe-{seq}.jsonl"
    lines = []
    for i in range(rows):
        body = " ".join(words[(i + k) % len(words)] for k in range(int(seq * 1.7)))
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"Story {i}?"},
                        {"role": "assistant", "content": body},
                    ]
                }
            )
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text or "gpu_oom" in text


def _child(payload: dict[str, Any]) -> int:  # pragma: no cover - GPU only (tests/test_vram_calibration_gpu.py)
    import gc

    import torch

    from .trainer import Trainer, estimate_vram

    model, mode, base_4bit = payload["model"], payload["mode"], bool(payload["base_4bit"])
    device = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(device)
    free_gib, total_gib = free_b / GIB, total_b / GIB
    plan_budget = probe_budget(free_gib, total_gib)
    if plan_budget is None:
        _emit({"event": "result", "error": (
            f"Only {free_gib:.1f} GB of GPU memory is free; close other GPU "
            "programs and try again.")})
        return 0
    budget_gib, fraction = plan_budget
    # The cap: an overrun raises a clean out-of-memory error instead of
    # spilling into shared system memory (which stalls the whole machine).
    # Never above what is free minus the headroom.
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    # The cap does not cover a paged optimizer's managed memory, so the
    # probes never use one (full mode forces it, and so does LoRA under
    # 24 GB). This is a probe-only process; the setting goes no further.
    Trainer._detect_optim_for_card = staticmethod(  # type: ignore[method-assign,assignment]
        lambda _configured: PROBE_OPTIM
    )

    # Before loading: is there room for the weights at all?
    kw_w: dict[str, Any] = {"mode": mode, "batch_size": 1, "max_seq_length": 256,
                            "use_calibration": False}
    if mode == "lora":
        kw_w.update(lora_r=PROBE_LORA_R, target_modules=list(PROBE_TARGETS),
                    quantize_base=base_4bit)
    estimate = estimate_vram(model, **kw_w)
    predicted_load = float(estimate.model_weights_gb)
    if mode == "full":
        # A full fine-tune also holds gradients and optimizer state.
        predicted_load += estimate.param_count_billions * 1e9 * FULL_FT_TRAIN_BYTES / GIB
    if predicted_load > budget_gib * PROBE_BUDGET_SHARE:
        _emit({"event": "result", "error": (
            f"Measuring this model is predicted to need at least {predicted_load:.1f} GB, "
            f"and {budget_gib:.1f} GB is available. Nothing was run.")})
        return 0

    _emit({"event": "loading", "budget_gib": round(budget_gib, 2)})
    tmp = Path(tempfile.mkdtemp(prefix="bp-vram-cal-"))
    kwargs: dict[str, Any] = {
        "model": model, "mode": mode, "batch_size": 1,
        "max_seq_length": max(s for _b, s in PROBE_CANDIDATES),
        "output_dir": str(tmp / "out"), "report_to": "none", "oom_recovery": False,
    }
    if mode == "lora":
        kwargs.update(lora_r=PROBE_LORA_R, lora_alpha=2 * PROBE_LORA_R,
                      target_modules=list(PROBE_TARGETS), load_in_4bit=base_4bit)
    try:
        trainer = Trainer(**kwargs)
        trainer.load_model()
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        reason = "ran out of GPU memory" if _is_oom(exc) else f"failed ({type(exc).__name__}: {exc})"
        _emit({"event": "result", "error": f"Loading {model} {reason}."[:500]})
        return 0
    torch.cuda.synchronize()
    load_gib = torch.cuda.memory_allocated() / GIB
    net = trainer._model
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    cfg = net.config
    hidden = int(cfg.hidden_size)
    # The floor: training makes one transient fp32 copy of the embedding table.
    floor_gib = int(net.get_input_embeddings().weight.numel()) * 4 / GIB
    # The formula's per-row coefficients for THIS model (see estimate_vram).
    # Linear in the row length: the fp32 logits plus per-token activations.
    formula_quad = 0.0
    formula_lin = 4.0 * int(cfg.vocab_size) + 30.0 * hidden
    _emit({"event": "loaded", "load_gib": round(load_gib, 2), "floor_gib": round(floor_gib, 2)})

    def predicted_rows(batch: int, seq: int) -> float:
        return batch * (formula_quad * seq * seq + formula_lin * seq) / GIB

    # Full fine-tuning: gradients + optimizer state come on top of every probe.
    fixed_pred = trainable * FULL_FT_TRAIN_BYTES / GIB if mode == "full" else 0.0

    # Informative probes: above the floor, largest sequence first (the model
    # was loaded for the largest), and only what is predicted to fit.
    scale = 1.0  # measured / predicted, from the probes so far (never below 1)
    probes: list[dict[str, Any]] = []
    plan = [c for c in PROBE_CANDIDATES if predicted_rows(*c) >= FLOOR_CLEARANCE * floor_gib]
    plan.sort(key=lambda c: (-c[1], c[0]))
    for batch, seq in plan:
        if len([p for p in probes if not p.get("oom")]) >= MAX_PROBES:
            break
        predicted = load_gib + fixed_pred + max(floor_gib, scale * predicted_rows(batch, seq))
        if predicted > budget_gib * PROBE_BUDGET_SHARE:
            _emit({"event": "skipped", "batch": batch, "seq": seq,
                   "predicted_gib": round(predicted, 2)})
            continue
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated() / GIB
        torch.cuda.reset_peak_memory_stats()
        trainer.batch_size = batch
        trainer.max_seq_length = seq
        row: dict[str, Any] = {"batch": batch, "seq": seq}
        try:
            trainer.train(dataset=str(_probe_dataset(tmp, seq, 4 * batch)), steps=PROBE_STEPS)
            peak = torch.cuda.max_memory_allocated() / GIB
            row.update(overhead_gib=round(peak - base, 4), peak_gib=round(peak, 3))
            if predicted_rows(batch, seq) > 0:
                scale = max(scale, (peak - base - fixed_pred) / predicted_rows(batch, seq))
        except Exception as exc:  # noqa: BLE001 - an OOM probe is a data point
            if not _is_oom(exc):
                if any(not p.get("oom") for p in probes):
                    # Keep what was measured before the failure.
                    _emit({"event": "skipped", "batch": batch, "seq": seq,
                           "reason": f"{type(exc).__name__}: {exc}"[:300]})
                    break
                _emit({"event": "result", "error": (
                    f"A probe failed ({type(exc).__name__}: {exc})."[:500])})
                return 0
            row["oom"] = True
        probes.append(row)
        _emit({"event": "probe", **row})
        if row.get("oom"):
            break
    _emit({"event": "result", "load_gib": round(load_gib, 4),
           "floor_gib": round(floor_gib, 4), "trainable_params": int(trainable),
           "formula_quad": formula_quad, "formula_lin": formula_lin, "probes": probes,
           "optim": PROBE_OPTIM})
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the GPU test
    raise SystemExit(_child(json.loads(sys.argv[1])))
