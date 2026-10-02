"""Measure training VRAM on the GPU that is running, and reuse the result.

``backprop estimate-vram <model> --calibrate`` (and "Measure on this GPU" in
the web UI) run a few very short REAL training probes of one model and fit
that model's memory cost on this machine:

    peak(batch, seq) = load + max(floor, batch * (quad * seq^2 + lin * seq))

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
  start. On Windows an uncapped overrun does not fail: the driver spills into
  shared system memory and the whole desktop stutters. Capped, it raises a
  clean out-of-memory error instead.
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
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = 1
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
#: GPU memory left untouched below what was free at start (GiB).
HEADROOM_GIB = 1.0
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
ADAPTER_BYTES_TRAINING = 6.3


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
    # Per row: bytes per token^2 and per token. None when no probe above the
    # floor fitted this card; the estimate then uses the formula's row cost.
    quad_bytes: float | None
    lin_bytes: float | None
    probe_trainable_params: int
    probes: list[dict[str, Any]] = field(default_factory=list)
    max_residual_pct: float = 0.0
    measured_at: str = ""
    version: int = CALIBRATION_VERSION

    @property
    def rows_measured(self) -> bool:
        return self.quad_bytes is not None and self.lin_bytes is not None

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
        total = torch.cuda.get_device_properties(0).total_memory / GIB
        name = torch.cuda.get_device_name(0)
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
        "varlen_attention": bool(_varlen_attention_installed()),
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


def _read_store() -> dict[str, Any]:
    path = calibration_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


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
        return Calibration(**raw)
    except TypeError:
        return None


def save(cal: Calibration) -> Path:
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store = _read_store()
    store[_key(cal.model, cal.mode, cal.base_4bit, cal.machine)] = asdict(cal)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


# ---- the fit -----------------------------------------------------------------------


def fit_rows(
    probes: list[dict[str, Any]], formula_quad: float, formula_lin: float
) -> tuple[float, float, float] | None:
    """Per-row ``(quad_bytes, lin_bytes, max_residual_pct)`` from the probes.

    ``probes`` rows carry ``batch``, ``seq`` and ``overhead_gib`` (peak minus
    the loaded baseline), all above the floor. With two or more sequence
    lengths the two terms are fitted (non-negative least squares); otherwise
    the formula's own split between them is scaled to match the measurement.
    None when nothing usable was measured.
    """
    import numpy as np

    rows = [p for p in probes if not p.get("oom") and p.get("overhead_gib")]
    if not rows:
        return None
    y = np.array([float(p["overhead_gib"]) * GIB for p in rows])
    quad_col = np.array([p["batch"] * float(p["seq"]) ** 2 for p in rows])
    lin_col = np.array([p["batch"] * float(p["seq"]) for p in rows])
    quad = lin = -1.0
    if len({p["seq"] for p in rows}) >= 2:
        sol, *_ = np.linalg.lstsq(np.stack([quad_col, lin_col], axis=1), y, rcond=None)
        quad, lin = float(sol[0]), float(sol[1])
    if quad < 0 or lin < 0:
        base = formula_quad * quad_col + formula_lin * lin_col
        scale = float(y.sum() / base.sum()) if base.sum() > 0 else 0.0
        if scale <= 0:
            return None
        quad, lin = formula_quad * scale, formula_lin * scale
    pred = quad * quad_col + lin * lin_col
    resid = float(np.max(np.abs(pred - y) / np.maximum(y, 1.0))) * 100.0
    return quad, lin, resid


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
    proc = subprocess.Popen(  # noqa: S603 - argv built here, no shell
        [sys.executable, "-m", "backpropagate.vram_calibration", json.dumps(payload)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    result: dict[str, Any] | None = None
    tail: list[str] = []
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
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
        if time.monotonic() > deadline:
            proc.kill()
            raise CalibrationError("Calibration timed out and was stopped.")
    rc = proc.wait()
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
    )
    save(cal)
    return cal


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

    from .trainer import Trainer, _varlen_attention_available, estimate_vram

    model, mode, base_4bit = payload["model"], payload["mode"], bool(payload["base_4bit"])
    free_b, total_b = torch.cuda.mem_get_info()
    free_gib, total_gib = free_b / GIB, total_b / GIB
    budget_gib = free_gib - HEADROOM_GIB
    if budget_gib <= 1.0:
        _emit({"event": "result", "error": (
            f"Only {free_gib:.1f} GB of GPU memory is free; close other GPU "
            "programs and try again.")})
        return 0
    # The cap: an overrun raises a clean out-of-memory error instead of
    # spilling into shared system memory (which stalls the whole machine).
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(0.98, budget_gib / total_gib)))

    # Before loading: is there room for the weights at all?
    kw_w: dict[str, Any] = {"mode": mode, "batch_size": 1, "max_seq_length": 256,
                            "use_calibration": False}
    if mode == "lora":
        kw_w.update(lora_r=PROBE_LORA_R, target_modules=list(PROBE_TARGETS),
                    quantize_base=base_4bit)
    predicted_load = float(estimate_vram(model, **kw_w).model_weights_gb)
    if predicted_load > budget_gib * PROBE_BUDGET_SHARE:
        _emit({"event": "result", "error": (
            f"Loading this model is predicted to need {predicted_load:.1f} GB, "
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
    heads = int(getattr(cfg, "num_attention_heads", 0) or 32)
    # The floor: training makes one transient fp32 copy of the embedding table.
    floor_gib = int(net.get_input_embeddings().weight.numel()) * 4 / GIB
    # The formula's per-row coefficients for THIS model (see estimate_vram).
    varlen = _varlen_attention_available(net, unsloth_loaded=bool(trainer.use_unsloth))
    formula_quad = 0.0 if (varlen or mode == "full") else 16.0 * heads
    formula_lin = 35.0 * hidden + (4.0 * int(cfg.vocab_size) if formula_quad == 0.0 else 0.0)
    _emit({"event": "loaded", "load_gib": round(load_gib, 2), "floor_gib": round(floor_gib, 2)})

    def predicted_rows(batch: int, seq: int) -> float:
        return batch * (formula_quad * seq * seq + formula_lin * seq) / GIB

    # Informative probes: above the floor, largest sequence first (the model
    # was loaded for the largest), and only what is predicted to fit.
    scale = 1.0  # measured / predicted, from the probes so far (never below 1)
    probes: list[dict[str, Any]] = []
    plan = [c for c in PROBE_CANDIDATES if predicted_rows(*c) >= FLOOR_CLEARANCE * floor_gib]
    plan.sort(key=lambda c: (-c[1], c[0]))
    for batch, seq in plan:
        if len([p for p in probes if not p.get("oom")]) >= MAX_PROBES:
            break
        predicted = load_gib + max(floor_gib, scale * predicted_rows(batch, seq))
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
                scale = max(scale, (peak - base) / predicted_rows(batch, seq))
        except Exception as exc:  # noqa: BLE001 - an OOM probe is a data point
            if not _is_oom(exc):
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
           "formula_quad": formula_quad, "formula_lin": formula_lin, "probes": probes})
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the GPU test
    raise SystemExit(_child(json.loads(sys.argv[1])))
