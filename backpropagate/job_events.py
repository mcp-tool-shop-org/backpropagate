"""UI job event plumbing — child (training-process) side.

ui-v2 P1. The web UI never trains in-process: it spawns
``python -m backpropagate train --ui-run-dir <run_dir>`` and observes the run
through files in ``run_dir``. This module is the writer half of that file
contract — it must stay importable WITHOUT Reflex installed (``[ui]`` is an
optional extra), so it lives at package root rather than under ``ui_app/``.

File contract (all paths inside ``run_dir``)::

    events.jsonl   child -> UI, append-only, one JSON object per line
    control.json   UI -> child, {"action": "stop" | "stop_save", "ts": ...}
    output.log     child stdout/stderr (opened by the parent, not here)
    job.json       spawn record (parent) + completion record (cli.py)

Event kinds written by this module:

- ``phase``      ``{ts, kind, phase}`` — phase in loading/training/saving/done.
- ``step``       ``{ts, kind, step, total_steps, phase, loss, ema_loss, lr,
                    step_time_ms, vram_alloc_gib, vram_reserved_gib,
                    vram_device_used_gib, vram_device_total_gib, temp_c}``
                  emitted from the HF Trainer ``on_log`` hook at the trainer's
                  logging cadence. ``ema_loss`` is the debiased EMA (beta 0.9);
                  both raw and smoothed are kept so the UI can render raw
                  faint + EMA bold (design digest). ``vram_alloc/reserved`` are
                  PER-PROCESS (torch); ``vram_device_*`` + ``temp_c`` are
                  device-wide via gpu_safety.get_system_gpu_readings (pynvml
                  when present, else the driver-bundled nvidia-smi).
- ``run``        ``{ts, kind, run, runs}`` — multi-run only: run ``run`` of
                  ``runs`` is starting. Step events then carry the session-wide
                  step (``(run - 1) * steps_per_run + local step``).
- ``checkpoint`` ``{ts, kind, path}`` — from ``on_save``.
- ``safety``     ``{ts, kind, reason}`` — a safety limit (``--gpu-max-temp``)
                  ended the run early; written by :meth:`JobEventWriter.safety`
                  when the limit trips.
- ``done``       ``{ts, kind, status, steps_done[, output_path][, reason]}`` —
                  terminal record, written by ``cli.py`` (which owns the
                  outcome) via :meth:`JobEventWriter.done`. ``reason`` is set
                  only for a ``stopped`` run that a safety limit ended.
- ``error``      ``{ts, kind, status, code, message, hint, traceback_tail}``.

The stop contract: ``control.json`` with a JSON object whose ``action`` is
``stop`` or ``stop_save`` flips ``should_training_stop`` AND ``should_save``
at the next ``on_step_end`` — "Stop and save checkpoint", never a hard kill.
A malformed control file is tolerated (ignored), never fatal to training.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVENTS_FILENAME = "events.jsonl"
CONTROL_FILENAME = "control.json"

#: control.json actions that request a cooperative stop-and-save.
CONTROL_STOP_ACTIONS = frozenset({"stop", "stop_save"})

#: Phase names, in lifecycle order.
PHASES = ("loading", "training", "saving", "done")

#: EMA smoothing for the debiased loss curve (design digest: raw faint,
#: EMA bold).
_EMA_BETA = 0.9


class JobEventWriter:
    """Append-only JSONL writer for ``<run_dir>/events.jsonl``.

    Thread-safe: ``on_log`` (callback thread) and the parent-side CLI
    completion write can race at run end.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / EVENTS_FILENAME
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._last_phase: str | None = None

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def write(self, event: dict[str, Any]) -> None:
        """Write one event row. Never raises into training — a UI-side disk
        problem must not kill a good run; it logs WARN instead."""
        row = dict(event)
        row.setdefault("ts", self._ts())
        try:
            line = json.dumps(row, default=str)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 — writer must never poison a run
            logger.warning("JobEventWriter.write failed: %r", exc)

    def phase(self, phase: str, **extra: Any) -> None:
        """Record a phase change. Re-entering the current phase is a no-op:
        a cooperative stop enters "saving" in the callback (checkpoint) and
        the CLI enters it again for the final save, and the UI should show
        one phase change, not two."""
        if phase == self._last_phase and not extra:
            return
        self._last_phase = phase
        self.write({"kind": "phase", "phase": phase, **extra})

    def run_marker(self, run: int, runs: int) -> None:
        """Multi-run: run ``run`` of ``runs`` (1-based) is starting."""
        self.write({"kind": "run", "run": int(run), "runs": int(runs)})

    def checkpoint(self, path: str | Path) -> None:
        self.write({"kind": "checkpoint", "path": str(path)})

    def done(
        self,
        *,
        status: str,
        steps_done: int,
        output_path: str | None = None,
        reason: str | None = None,
    ) -> None:
        row: dict[str, Any] = {"kind": "done", "status": status, "steps_done": steps_done}
        if output_path:
            row["output_path"] = output_path
        if reason:
            # Why a "stopped" run stopped when it was not the operator's Stop
            # (e.g. the --gpu-max-temp limit tripped).
            row["reason"] = reason
        self.write(row)

    def safety(self, reason: str) -> None:
        """A safety limit ended the run early (e.g. GPU over ``--gpu-max-temp``).

        Written the moment the limit trips; the terminal ``done`` event that
        follows carries ``status="stopped"`` and the same ``reason``.
        """
        self.write({"kind": "safety", "reason": str(reason)})

    def error(
        self,
        *,
        code: str,
        message: str,
        hint: str | None = None,
        traceback_tail: str | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "kind": "error",
            "status": "failed",
            "code": code,
            "message": message,
        }
        if hint:
            row["hint"] = hint
        if traceback_tail:
            row["traceback_tail"] = traceback_tail
        self.write(row)


def read_stop_request(run_dir: str | Path) -> bool:
    """True when ``control.json`` exists with a stop action; False on any
    missing/malformed content (never raises)."""
    control_path = Path(run_dir) / CONTROL_FILENAME
    try:
        if not control_path.exists():
            return False
        payload = json.loads(control_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        return payload.get("action") in CONTROL_STOP_ACTIONS
    except Exception:  # noqa: BLE001 — malformed control file must never crash training
        return False


class _CallbackBase:  # pragma: no cover - exercised via the real subclass
    """Fallback base when transformers is unavailable (keeps import cheap)."""


_BASE: type
try:  # transformers is a hard dep; the fallback is for exotic import orders.
    from transformers import TrainerCallback as _TrainerCallback

    _BASE = _TrainerCallback
except Exception:  # noqa: BLE001  # nosec B110 — degraded import fallback
    _BASE = _CallbackBase  # type: ignore[assignment]


class UiFileEventCallback(_BASE):  # type: ignore[misc, valid-type]
    """HF ``TrainerCallback`` that mirrors training progress to files.

    Args:
        run_dir: The ``--ui-run-dir`` directory.
        writer: Optional pre-built :class:`JobEventWriter` (tests).
        gpu_poll_s: Minimum seconds between ``get_gpu_status()`` temperature
            polls (NVML is system-wide but not free). VRAM figures come from
            ``torch.cuda.memory_*`` every logged step.
        total_steps: Overall step total to report. ``None`` uses the HF
            ``state.max_steps`` (single run). Multi-run passes
            ``runs * steps_per_run`` so the UI shows one bar for the session.
        on_stop: Called once when a stop is requested. Multi-run passes
            ``MultiRunTrainer.abort`` so the run loop ends after the current
            run unwinds, instead of starting the next run.

    ``step_offset`` (attribute) is added to every reported step. Multi-run
    sets it to ``(run - 1) * steps_per_run`` at each run start, because the
    inner HF ``global_step`` restarts at 0 for every run.
    """

    def __init__(
        self,
        run_dir: str | Path,
        writer: JobEventWriter | None = None,
        gpu_poll_s: float = 1.0,
        total_steps: int | None = None,
        on_stop: Any = None,
    ) -> None:
        super().__init__()
        self.total_steps = total_steps
        self.on_stop = on_stop
        self.step_offset: int = 0
        self.run_dir = Path(run_dir)
        self.writer = writer if writer is not None else JobEventWriter(self.run_dir)
        self._gpu_poll_s = float(gpu_poll_s)
        self._last_gpu_poll = 0.0
        self._last_temp_c: float | None = None
        self._last_sys_used_gib: float | None = None
        self._last_sys_total_gib: float | None = None
        self._last_log_time: float | None = None
        self._last_log_step: int = 0
        self._ema: float = 0.0
        self._ema_n: int = 0
        # Highest global_step ever observed (on_step_end fires EVERY step, so
        # this is exact even when logging_steps > 1). cli.py reads it for the
        # terminal `done` event's steps_done so a stopped run reports the step
        # it actually reached, not the requested total.
        self.last_step: int = 0
        # One-shot guard: control.json persists after the stop request, so
        # without this flag every later on_step_end re-emitted phase("saving").
        self._stop_signaled: bool = False

    # ---- loss EMA ----------------------------------------------------------

    def _push_ema(self, loss: float | None) -> float | None:
        """Debiased EMA (beta 0.9); returns the debiased value or None.

        The accumulator starts at 0 so the bias correction
        ``acc / (1 - beta**n)`` is exact from the first observation.
        """
        if loss is None:
            if self._ema_n == 0:
                return None
        else:
            self._ema_n += 1
            self._ema = _EMA_BETA * self._ema + (1.0 - _EMA_BETA) * loss
        debias = 1.0 - _EMA_BETA**self._ema_n
        return self._ema / debias if debias > 0 else self._ema

    # ---- GPU snapshot --------------------------------------------------------

    def _gpu_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "vram_alloc_gib": None,
            "vram_reserved_gib": None,
            "vram_device_used_gib": self._last_sys_used_gib,
            "vram_device_total_gib": self._last_sys_total_gib,
            "temp_c": None,
        }
        try:
            import torch

            if torch.cuda.is_available():
                gib = 1024**3
                fields["vram_alloc_gib"] = round(torch.cuda.memory_allocated() / gib, 2)
                fields["vram_reserved_gib"] = round(torch.cuda.memory_reserved() / gib, 2)
        except Exception:  # noqa: BLE001  # nosec B110 — telemetry must never kill training
            pass
        now = time.time()
        if now - self._last_gpu_poll >= self._gpu_poll_s:
            self._last_gpu_poll = now
            try:
                from .gpu_safety import get_system_gpu_readings

                # DEVICE-WIDE readings (temp + whole-card VRAM) via pynvml →
                # nvidia-smi. pynvml is not a hard dep, which is exactly why
                # temp_c used to stay null; the nvidia-smi fallback is the
                # path that actually works on a stock install.
                readings = get_system_gpu_readings()
                if readings is not None:
                    self._last_temp_c = readings.temperature_c
                    self._last_sys_used_gib = readings.memory_used_gib
                    self._last_sys_total_gib = readings.memory_total_gib
            except Exception:  # noqa: BLE001 — temperature is optional telemetry
                self._last_temp_c = None
        fields["temp_c"] = self._last_temp_c
        fields["vram_device_used_gib"] = self._last_sys_used_gib
        fields["vram_device_total_gib"] = self._last_sys_total_gib
        return fields

    # ---- HF lifecycle hooks --------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        self.writer.phase("training")
        self._last_log_time = time.time()
        self._last_log_step = 0

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> Any:  # noqa: ARG002
        logs = logs or {}
        step = int(getattr(state, "global_step", 0) or 0)
        logging_steps = getattr(args, "logging_steps", 1) or 1
        if logging_steps != 1 and step % logging_steps != 0:
            return
        now = time.time()
        if self._last_log_time is not None and step > self._last_log_step:
            elapsed_ms = (now - self._last_log_time) * 1000.0
            step_time_ms: float | None = elapsed_ms / (step - self._last_log_step)
        else:
            step_time_ms = None
        self._last_log_time = now
        self._last_log_step = step
        loss = logs.get("loss")
        try:
            loss = float(loss) if loss is not None else None
        except (TypeError, ValueError):
            loss = None
        lr = logs.get("learning_rate")
        try:
            lr = float(lr) if lr is not None else None
        except (TypeError, ValueError):
            lr = None
        row: dict[str, Any] = {
            "kind": "step",
            "step": self.step_offset + step,
            "total_steps": int(
                self.total_steps or getattr(state, "max_steps", 0) or 0
            ),
            "phase": "training",
            "loss": loss,
            "ema_loss": self._push_ema(loss),
            "lr": lr,
            "step_time_ms": step_time_ms,
            "epoch": getattr(state, "epoch", None),
        }
        row.update(self._gpu_fields())
        self.writer.write(row)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        step = self.step_offset + int(getattr(state, "global_step", 0) or 0)
        if step > self.last_step:
            self.last_step = step
        if read_stop_request(self.run_dir):
            # "Stop and save checkpoint": finish the current step cleanly,
            # write the checkpoint, then unwind the training loop. Emit the
            # "saving" phase ONCE — control.json is not consumed/deleted, so
            # an unguarded emit repeats on every remaining step boundary.
            control.should_training_stop = True
            control.should_save = True
            if not self._stop_signaled:
                self._stop_signaled = True
                self.writer.phase("saving")
                if self.on_stop is not None:
                    try:
                        self.on_stop("Stopped from the web UI")
                    except Exception as exc:  # noqa: BLE001 — the HF stop above still applies
                        logger.warning("UI stop hook failed: %r", exc)
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        output_dir = getattr(args, "output_dir", None)
        step = int(getattr(state, "global_step", 0) or 0)
        if output_dir:
            self.writer.checkpoint(os.path.join(str(output_dir), f"checkpoint-{step}"))


__all__ = [
    "CONTROL_FILENAME",
    "CONTROL_STOP_ACTIONS",
    "EVENTS_FILENAME",
    "PHASES",
    "JobEventWriter",
    "UiFileEventCallback",
    "read_stop_request",
]
