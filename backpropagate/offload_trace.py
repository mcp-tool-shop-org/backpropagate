"""Per-leg timing and byte counters for the offload engine (``BACKPROPAGATE_OFFLOAD_TRACE``).

Engine A (:mod:`backpropagate.offload_engine`) is transfer-bound: at 7.6B it
moves about 91 GB over PCIe per step and takes 14.7 s. Before changing the
transfer pattern, this module measures where the time and the bytes go, so a
pod receipt can say which leg an optimization moved.

Modes, read from ``BACKPROPAGATE_OFFLOAD_TRACE``:

* unset / ``0`` / ``off``: no tracing. No CUDA events, no sync, no allocation,
  and no patched FSDP2 functions. The engine holds ``None`` and branches on it.
* ``1`` (any other value): per-leg timing with CUDA events plus byte counters.
* ``profile``: the above, and ``torch.profiler`` runs over one step (the second,
  so lazy init is out of the trace). Its summary lists the memcpy events by
  direction (count, bytes, GPU time) and the ``FSDP::`` ranges.

The legs, per optimizer step:

* ``forward`` / ``backward``: the whole phase. ``backward`` includes the
  activation-checkpoint recompute, the backward re-gather and the gradient
  copy to the host. With the fused optimizer (``BACKPROPAGATE_OFFLOAD_FUSED``)
  it also includes the optimizer legs, because they run inside backward.
* ``fwd_gather`` / ``bwd_gather``: time the compute stream spends in FSDP2's
  ``wait_for_unshard`` for a group whose all-gather is pending, with the bytes
  of that group's host shard. This is the exposed (not overlapped) part of the
  host-to-device parameter copy.
* ``grad_reduce_d2h``: FSDP2's ``foreach_reduce`` (reduce-scatter copy-in, the
  single-rank copy, and the gradient copy to the host), with the gradient
  bytes. It reads 0 bytes when the fused optimizer consumed the gradients first.
* ``opt_total`` / ``opt_h2d`` / ``opt_writeback_d2h``: the optimizer's work per
  parameter, the host-to-device copies it makes (gradient chunks and weights),
  and the write-back of the updated weights. ``opt_compute`` is derived as
  ``opt_total - opt_h2d - opt_writeback_d2h``.

CUDA events measure the time between two points of the current stream, so a
leg that waits on another stream includes the wait, and nested legs overlap
(``backward`` contains ``grad_reduce_d2h``). Legs are not meant to add up to
the step time. ``step_ms`` is the wall clock of the whole step.

The FSDP2 probes wrap private functions of ``torch.distributed.fsdp``
(``FSDPParamGroup.wait_for_unshard`` and ``foreach_reduce``). They exist only
while tracing, are restored on exit, and are skipped with a warning if torch
does not have them. They never change what the engine computes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

_OFF_VALUES = {"", "0", "off", "false", "no"}

# The derived leg is reported next to the measured ones.
_COMPUTE_LEG = "opt_compute"


def trace_mode() -> str:
    """``"off"``, ``"legs"`` or ``"profile"`` from ``BACKPROPAGATE_OFFLOAD_TRACE``."""
    raw = os.environ.get("BACKPROPAGATE_OFFLOAD_TRACE", "").strip().lower()
    if raw in _OFF_VALUES:
        return "off"
    return "profile" if raw == "profile" else "legs"


class _Leg:
    """Context manager that times one leg with CUDA events (or the wall clock on CPU)."""

    __slots__ = ("_name", "_nbytes", "_owner", "_start")

    def __init__(self, owner: LegTrace, name: str, nbytes: int) -> None:
        self._owner = owner
        self._name = name
        self._nbytes = nbytes
        self._start: Any = None

    def __enter__(self) -> _Leg:
        if self._owner.cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        owner = self._owner
        if owner.cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            owner.pending.append((self._name, self._start, end, self._nbytes))
        else:
            owner.add(self._name, (time.perf_counter() - self._start) * 1e3, self._nbytes)


class LegTrace:
    """Collects per-leg milliseconds and bytes, folded into one record per step."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.pending: list[tuple[str, Any, Any, int]] = []
        self._ms: dict[str, float] = {}
        self._bytes: dict[str, int] = {}
        self._calls: dict[str, int] = {}
        self.steps: list[dict[str, Any]] = []
        self.profile: dict[str, Any] | None = None

    def leg(self, name: str, nbytes: int = 0) -> _Leg:
        return _Leg(self, name, int(nbytes))

    def add(self, name: str, ms: float, nbytes: int = 0) -> None:
        self._ms[name] = self._ms.get(name, 0.0) + ms
        self._bytes[name] = self._bytes.get(name, 0) + nbytes
        self._calls[name] = self._calls.get(name, 0) + 1

    def count_bytes(self, name: str, nbytes: int) -> None:
        """Bytes for a leg that has no timer of its own."""
        self._bytes[name] = self._bytes.get(name, 0) + int(nbytes)

    def end_step(self, step: int, step_ms: float) -> dict[str, Any]:
        """Fold this step's events into a record. Call after the step's final sync."""
        if self.cuda:
            torch.cuda.synchronize()
            for name, start, end, nbytes in self.pending:
                self.add(name, start.elapsed_time(end), nbytes)
        self.pending.clear()
        ms = dict(self._ms)
        if "opt_total" in ms:
            ms[_COMPUTE_LEG] = ms["opt_total"] - ms.get("opt_h2d", 0.0) - ms.get("opt_writeback_d2h", 0.0)
        record = {
            "step": step,
            "step_ms": round(step_ms, 2),
            "ms": {k: round(v, 2) for k, v in sorted(ms.items())},
            "bytes": dict(sorted(self._bytes.items())),
            "calls": dict(sorted(self._calls.items())),
        }
        self.steps.append(record)
        self._ms.clear()
        self._bytes.clear()
        self._calls.clear()
        return record

    def summary(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """The dict stored under ``"trace"`` in ``run_offload_training``'s result.

        ``mean`` averages the steps after the first (the first pays lazy init,
        CUDA context and allocator warm-up); with a single step it is that step.
        """
        steady = self.steps[1:] or self.steps
        mean_ms: dict[str, float] = {}
        mean_bytes: dict[str, float] = {}
        for rec in steady:
            for k, v in rec["ms"].items():
                mean_ms[k] = mean_ms.get(k, 0.0) + v / len(steady)
            for k, v in rec["bytes"].items():
                mean_bytes[k] = mean_bytes.get(k, 0.0) + v / len(steady)
        gb_per_s = {
            k: round(mean_bytes[k] / 1e9 / (mean_ms[k] / 1e3), 2)
            for k in mean_bytes
            if mean_bytes[k] > 0 and mean_ms.get(k, 0.0) > 0
        }
        return {
            "mode": "profile" if self.profile is not None else "legs",
            "config": config or {},
            "device": "cuda" if self.cuda else "cpu",
            "steps": self.steps,
            "mean_over_steps": len(steady),
            "mean_step_ms": round(sum(r["step_ms"] for r in steady) / max(1, len(steady)), 2),
            "mean_ms": {k: round(v, 2) for k, v in sorted(mean_ms.items())},
            "mean_bytes": {k: int(v) for k, v in sorted(mean_bytes.items())},
            "mean_gb_per_s": dict(sorted(gb_per_s.items())),
            "profile": self.profile,
        }


def _group_bytes(group: Any) -> int:
    """Host bytes of an FSDP2 parameter group (what a gather copies up)."""
    total = 0
    for fp in getattr(group, "fsdp_params", []) or []:
        data = getattr(fp, "_sharded_param_data", None)
        if data is not None:
            total += data.numel() * data.element_size()
    return total


def install_fsdp_probes(trace: LegTrace) -> Callable[[], None]:
    """Time FSDP2's gather wait and gradient reduce; returns a function that undoes it.

    Returns a no-op (after a warning) when this torch lacks the private names.
    """
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

        group_cls = pg.FSDPParamGroup
        orig_wait = group_cls.wait_for_unshard
        orig_reduce = pg.foreach_reduce
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "offload trace: FSDP2 gather/reduce probes unavailable (%s); "
            "fwd_gather, bwd_gather and grad_reduce_d2h will be missing.",
            exc,
        )
        return lambda: None

    def wait_for_unshard(self: Any) -> Any:
        if getattr(self, "_all_gather_result", None) is None:
            return orig_wait(self)  # nothing pending: FSDP2 returns at once
        state = getattr(getattr(self, "_training_state", None), "name", "")
        leg = "fwd_gather" if state == "FORWARD" else "bwd_gather"
        with trace.leg(leg, _group_bytes(self)):
            return orig_wait(self)

    def foreach_reduce(fsdp_params: Any, unsharded_grads: Any, *args: Any, **kwargs: Any) -> Any:
        # foreach_reduce clears the list it is given; count first.
        nbytes = sum(g.numel() * g.element_size() for g in unsharded_grads)
        with trace.leg("grad_reduce_d2h", nbytes):
            return orig_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    group_cls.wait_for_unshard = wait_for_unshard  # type: ignore[method-assign]
    pg.foreach_reduce = foreach_reduce

    def undo() -> None:
        group_cls.wait_for_unshard = orig_wait  # type: ignore[method-assign]
        pg.foreach_reduce = orig_reduce

    return undo


def summarize_chrome_trace(events: list[dict[str, Any]], top_kernels: int = 8) -> dict[str, Any]:
    """Reduce a Chrome-trace event list to memcpy, kernel and ``FSDP::`` totals.

    Kineto records each memcpy with its byte count, so this gives the measured
    transfer volume per direction for the profiled step.
    """
    memcpy: dict[str, dict[str, float]] = {}
    kernels: dict[str, dict[str, float]] = {}
    ranges: dict[str, dict[str, float]] = {}
    for ev in events:
        cat, name, dur = ev.get("cat"), str(ev.get("name", "")), float(ev.get("dur", 0.0))
        if cat in ("gpu_memcpy", "Memcpy"):
            slot = memcpy.setdefault(name, {"count": 0, "ms": 0.0, "bytes": 0})
            slot["count"] += 1
            slot["ms"] += dur / 1e3
            slot["bytes"] += int((ev.get("args") or {}).get("bytes", 0))
        elif cat in ("kernel", "Kernel"):
            slot = kernels.setdefault(name, {"count": 0, "ms": 0.0})
            slot["count"] += 1
            slot["ms"] += dur / 1e3
        elif cat == "user_annotation" and name.startswith("FSDP::"):
            slot = ranges.setdefault(name, {"count": 0, "ms": 0.0})
            slot["count"] += 1
            slot["ms"] += dur / 1e3

    def rounded(d: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        return {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sorted(d.items())}

    top = dict(sorted(kernels.items(), key=lambda kv: kv[1]["ms"], reverse=True)[:top_kernels])
    return {
        "memcpy": rounded(memcpy),
        "kernel_ms_total": round(sum(v["ms"] for v in kernels.values()), 3),
        "kernel_top": rounded(top),
        "fsdp_ranges_cpu_ms": rounded(ranges),
    }


@contextlib.contextmanager
def profile_step(trace: LegTrace) -> Iterator[None]:
    """Run ``torch.profiler`` over the wrapped step and store its summary on ``trace``."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if trace.cuda:
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        yield
    if trace.cuda:
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json", prefix="offload_trace_")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8") as fh:
            events = json.load(fh).get("traceEvents", [])
        trace.profile = summarize_chrome_trace(events)
    finally:
        os.unlink(path)
