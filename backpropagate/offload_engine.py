"""Single-card full fine-tuning with FSDP2 CPU offload (``full_ft_offload=True``).

The v1.7 path went through ``TrainingArguments(fsdp=...)`` / accelerate, which
upcasts every trainable parameter to fp32 and pairs it with torch AdamW. On the
host that costs ~16 bytes/param: fp32 params (4) + fp32 grads (4) + two fp32 Adam
moments (8). That is about 113 GB for a 7.6B model, so it cannot fit a 64 GB host.

This engine applies FSDP2 directly and keeps the host footprint at ~4 B/param:

* **Parameters stay in the load dtype (bf16).** We call ``fully_shard`` per decoder
  layer + the root ourselves, with ``MixedPrecisionPolicy(param_dtype=bf16)`` and
  ``CPUOffloadPolicy``. The params are not upcast, so the host holds bf16 params
  (2 B/param) and bf16 grads (2 B/param).
* **The optimizer is Adafactor with factored second moments and no first
  moment** (Shazeer & Stern, 2018). Its state is one row vector and one column
  vector per matrix, a few MB for a 7B model. That state lives on the GPU, so it
  adds ~0 B/param on the host.
* **The optimizer step runs on the GPU.** Each param shard and its grad are
  streamed to the card, updated in fp32, and written back. The CPU step was the
  bottleneck: measured 3-11 s/step at 360M, against 0.14 s on the GPU.
* **Updates are written back with stochastic rounding.** bf16 has an 8-bit
  mantissa, so at a full-FT learning rate (2e-5) round-to-nearest drops most
  updates: only 2.99 % of params changed after step 1, measured. Stochastic
  rounding keeps every update in expectation at zero extra bytes. Each
  parameter draws its rounding noise from its own seeded generator.
* **Optionally the step is fused into backward** (``BACKPROPAGATE_OFFLOAD_FUSED``,
  :class:`FusedBackwardStep`): each parameter is updated while its weights and
  gradient are already on the GPU, so the gradient never goes down to the host
  and the optimizer does not re-upload it. Same math, same routine.
* **Per-leg timing** is available with ``BACKPROPAGATE_OFFLOAD_TRACE``
  (:mod:`backpropagate.offload_trace`).

Trade-offs that callers and docs must state:
* Adafactor, not AdamW (no momentum; the second moment is factored).
* There are no fp32 master weights. Precision comes from stochastic rounding.
* The loop is plain causal-LM SFT over the whole sequence. It has no packing,
  no response-only masking, and no intermediate checkpoints (save() at the end).
* It needs Linux / WSL2 with NCCL (FSDP2); Windows-native fails fast with
  DEP_FSDP_UNAVAILABLE.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import random
import time
from collections.abc import Callable
from typing import Any

import torch

from .offload_trace import LegTrace, install_fsdp_probes, profile_step, trace_mode

logger = logging.getLogger(__name__)

# What a timing leg is when tracing is off: one shared, empty context manager.
_NO_LEG = contextlib.nullcontext()

# Rows of a 2-D parameter processed per optimizer chunk on the GPU. Bounds the
# transient fp32 working set of the step (a 7B embedding is 545M elements).
_MAX_CHUNK_NUMEL = 1 << 26


def _local(t: Any) -> Any:
    """The local shard of a DTensor, or the tensor itself."""
    to_local = getattr(t, "to_local", None)
    return to_local() if callable(to_local) else t


def stochastic_round_to_bf16_(
    dst: torch.Tensor, src_fp32: torch.Tensor, generator: torch.Generator | None = None
) -> None:
    """Write fp32 ``src_fp32`` into bf16 ``dst`` with stochastic rounding.

    Adds uniform noise below bf16's lowest kept bit, then truncates. The
    expected value of the result equals the fp32 input, so tiny updates survive
    on average instead of rounding to zero. ``generator`` (on ``src_fp32``'s
    device) makes the noise a function of its own state, not of the global RNG.
    """
    bits = src_fp32.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device, generator=generator)
    rounded = (bits + noise) & -65536
    dst.copy_(rounded.view(torch.float32))


class OffloadAdafactor(torch.optim.Optimizer):
    """Factored Adafactor (no momentum) that steps CPU-resident params on the GPU.

    For each parameter the grad and weight shard are moved to ``device`` in
    row chunks. The math is fp32, following Shazeer & Stern 2018:
    beta2_t = 1 - t^-0.8, a factored second moment, and update clipping to an
    RMS of ``clip_threshold``. The absolute lr applies the same way AdamW's
    does. The result is written back to the host with stochastic rounding for
    bf16 params and an exact copy for fp32. Row/column statistics live on the
    GPU.

    There is one update routine, :meth:`_update_param`. :meth:`step` calls it
    with the gradient on the host (the gradient FSDP2 copied down), and
    :meth:`step_param_fused` calls it from a backward hook with the gradient
    and weights already on the device, where the copies in it are no-ops. The
    two paths therefore run the same math. Each parameter also owns a
    ``torch.Generator`` for its stochastic rounding, seeded from ``seed`` and
    the parameter's position, so the noise does not depend on the order in
    which parameters are stepped (forward order here, backward order fused).
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        *,
        weight_decay: float = 0.0,
        beta2_decay: float = -0.8,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        stochastic_rounding: bool = True,
        device: torch.device | str | None = None,
        seed: int = 0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.beta2_decay = beta2_decay
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.stochastic_rounding = stochastic_rounding
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self.seed = seed
        self.last_update_retention: float | None = None
        self.last_fused_params = 0  # parameters the previous step() found already stepped in backward
        self.trace: LegTrace | None = None  # set by the train loop when tracing
        self._kept: torch.Tensor | None = None
        self._intended: torch.Tensor | None = None
        self._fused_done: set[int] = set()
        self._gens: dict[int, torch.Generator] = {}
        self._ordinal = {id(q): i for i, q in enumerate(q for g in self.param_groups for q in g["params"])}

    def _leg(self, name: str, nbytes: int = 0) -> Any:
        """A timing context for ``name`` when tracing, else an empty one."""
        return self.trace.leg(name, nbytes) if self.trace is not None else _NO_LEG

    def _fetch(self, src: torch.Tensor) -> torch.Tensor:
        """``src`` on the optimizer's device: a non-blocking copy, or ``src`` if it is already there."""
        dev = self.device
        if src.device.type == dev.type and (dev.index is None or src.device.index == dev.index):
            return src
        with self._leg("opt_h2d", src.numel() * src.element_size()):
            return src.to(dev, non_blocking=True)

    def _generator(self, p: torch.Tensor) -> torch.Generator:
        """The stochastic-rounding generator of ``p``, created on first use."""
        gen = self._gens.get(id(p))
        if gen is None:
            ordinal = self._ordinal.setdefault(id(p), len(self._ordinal))
            gen = torch.Generator(device=self.device)
            gen.manual_seed((self.seed + 1_000_003 * ordinal) & ((1 << 63) - 1))
            self._gens[id(p)] = gen
        return gen

    def _accumulators(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's (kept, intended) update sums, created on the device on first use."""
        if self._kept is None or self._intended is None:
            self._kept = torch.zeros((), dtype=torch.float64, device=self.device)
            self._intended = torch.zeros((), dtype=torch.float64, device=self.device)
        return self._kept, self._intended

    def _write_back(
        self,
        host: torch.Tensor,
        new_fp32: torch.Tensor,
        old_fp32: torch.Tensor,
        gen: torch.Generator | None = None,
    ) -> None:
        """Round ``new_fp32`` into ``host`` and account how much of the update survived.

        ``update_retention`` = sum(actual_delta * sign(intended)) / sum(|intended|):
        ~1.0 means the applied bf16 update equals the fp32 update in expectation.
        Round-to-nearest drops sub-ulp updates, and the ratio falls well below 1.
        """
        kept, intended_sum = self._accumulators()
        if host.dtype == torch.bfloat16:
            tmp = torch.empty_like(new_fp32, dtype=torch.bfloat16)
            if self.stochastic_rounding:
                stochastic_round_to_bf16_(tmp, new_fp32, gen)
            else:
                tmp.copy_(new_fp32)
            intended = new_fp32 - old_fp32
            kept += ((tmp.float() - old_fp32) * intended.sign()).sum()
            intended_sum += intended.abs().sum()
            with self._leg("opt_writeback_d2h", tmp.numel() * tmp.element_size()):
                host.copy_(tmp)
        else:
            with self._leg("opt_writeback_d2h", new_fp32.numel() * new_fp32.element_size()):
                host.copy_(new_fp32)
            delta = (new_fp32 - old_fp32).abs().sum()
            kept += delta
            intended_sum += delta

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:  # type: ignore[override]
        """Step every parameter that has a host gradient and was not already stepped in backward."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or id(p) in self._fused_done:
                    continue
                with self._leg("opt_total"):
                    self._update_param(p, lr, wd, _local(p), _local(p.grad))
        self.last_fused_params = len(self._fused_done)
        self._fused_done.clear()
        kept = float(self._kept) if self._kept is not None else 0.0
        intended = float(self._intended) if self._intended is not None else 0.0
        self._kept = self._intended = None
        self.last_update_retention = kept / intended if intended > 0 else 1.0
        return loss

    @torch.no_grad()
    def step_param_fused(
        self, p: torch.Tensor, grad: torch.Tensor, weight: torch.Tensor | None = None
    ) -> None:
        """Step ``p`` now, from a gradient (and optionally weights) already on the device.

        Called from a backward hook. ``step()`` later skips ``p`` and only
        finalizes the update-retention figure. Valid for one backward per
        optimizer step: a second call before ``step()`` means the gradient was
        accumulated over micro-batches, which a per-backward update cannot do.
        """
        if id(p) in self._fused_done:
            raise RuntimeError(
                "the fused backward step ran twice for one parameter in a single optimizer step; "
                "it needs gradient_accumulation == 1"
            )
        group = next((g for g in self.param_groups if any(q is p for q in g["params"])), None)
        if group is None:
            raise KeyError("step_param_fused: the parameter is not in any of the optimizer's groups")
        with self._leg("opt_total"):
            self._update_param(p, group["lr"], group["weight_decay"], _local(p), grad, weight)
        self._fused_done.add(id(p))

    def _update_param(
        self,
        p: torch.Tensor,
        lr: float,
        wd: float,
        w_host: torch.Tensor,
        g_src: torch.Tensor,
        w_src: torch.Tensor | None = None,
    ) -> None:
        """One Adafactor update of ``p``: grad ``g_src`` in, new weights into ``w_host``.

        ``g_src`` and ``w_src`` (default: ``w_host``) may be on the host, where
        each chunk is copied up, or already on the device, where the copy is a
        no-op. Neither is modified. The clip factor stays on the device, so the
        update does not wait on the GPU once per parameter.
        """
        dev = self.device
        gen = self._generator(p)
        st = self.state[p]
        if not st:
            st["step"] = 0
            if w_host.dim() >= 2:
                st["row"] = torch.zeros(w_host.shape[0], dtype=torch.float32, device=dev)
                st["col"] = torch.zeros(w_host[0].numel(), dtype=torch.float32, device=dev)
            else:
                st["v"] = torch.zeros(w_host.shape, dtype=torch.float32, device=dev)
        st["step"] += 1
        beta2 = 1.0 - st["step"] ** self.beta2_decay
        w_read = w_host if w_src is None else w_src

        if w_host.dim() < 2:
            g = self._fetch(g_src).float()
            st["v"].mul_(beta2).add_(g * g + self.eps, alpha=1.0 - beta2)
            u = g / st["v"].sqrt()
            u.div_(torch.clamp(u.pow(2).mean().sqrt() / self.clip_threshold, min=1.0))
            old = self._fetch(w_read).float()
            w = old.clone()
            if wd:
                w.mul_(1.0 - lr * wd)
            w.sub_(u, alpha=lr)
            self._write_back(w_host, w, old, gen)
            return

        rows = w_host.shape[0]
        g2d = g_src.reshape(rows, -1)
        w2d_host = w_host.view(rows, -1)  # a VIEW: writes must land in the param
        w2d_read = w_read.reshape(rows, -1)
        cols = g2d.shape[1]
        chunk = max(1, _MAX_CHUNK_NUMEL // max(1, cols))
        spans = [(i, min(rows, i + chunk)) for i in range(0, rows, chunk)]
        cached: dict[str, torch.Tensor] = {}  # single-chunk params keep g on the device

        def _g(
            a: int,
            b: int,
            src: torch.Tensor = g2d,
            single: bool = len(spans) == 1,
            cache: dict[str, torch.Tensor] = cached,
        ) -> torch.Tensor:
            if single:
                if "g" not in cache:
                    cache["g"] = self._fetch(src[a:b]).float()
                return cache["g"]
            return self._fetch(src[a:b]).float()

        # Pass 1: row / column means of g^2 -> factored second moment.
        col_sum = torch.zeros(cols, dtype=torch.float32, device=dev)
        row_mean = torch.empty(rows, dtype=torch.float32, device=dev)
        for a, b in spans:
            g2 = _g(a, b).pow(2).add_(self.eps)
            row_mean[a:b] = g2.mean(dim=1)
            col_sum.add_(g2.sum(dim=0))
        st["row"].mul_(beta2).add_(row_mean, alpha=1.0 - beta2)
        st["col"].mul_(beta2).add_(col_sum / rows, alpha=1.0 - beta2)
        row_norm = st["row"] / st["row"].mean()
        col_rsqrt = st["col"].rsqrt()

        # Pass 2: RMS of the update, for clipping. The factor is computed in
        # float64 on the device and rounded to fp32 once, as a Python float was.
        sumsq = torch.zeros((), dtype=torch.float32, device=dev)
        for a, b in spans:
            u = _g(a, b) * row_norm[a:b].rsqrt().unsqueeze(1) * col_rsqrt
            sumsq.add_(u.pow(2).sum())
        rms = (sumsq / (rows * cols)).sqrt()
        scale = (lr / torch.clamp(rms.double() / self.clip_threshold, min=1.0)).float()

        # Pass 3: apply and write back.
        for a, b in spans:
            u = _g(a, b) * row_norm[a:b].rsqrt().unsqueeze(1) * col_rsqrt
            u.mul_(scale)
            old = self._fetch(w2d_read[a:b]).float()
            w = old.clone()
            if wd:
                w.mul_(1.0 - lr * wd)
            w.sub_(u)
            self._write_back(w2d_host[a:b], w, old, gen)


def fused_backward_requested() -> bool:
    """``BACKPROPAGATE_OFFLOAD_FUSED``: step each parameter inside backward (default off)."""
    return os.environ.get("BACKPROPAGATE_OFFLOAD_FUSED", "").strip().lower() in {"1", "true", "yes", "on"}


class FusedBackwardStep:
    """Steps each parameter during backward, from the copy FSDP2 already has on the GPU.

    Per step the 3-pass path moves a parameter's bytes over PCIe like this: up
    for the forward gather, up again for the backward gather, down as the
    gradient (FSDP2's ``post_backward``), then up as the gradient (once per
    pass, so three times for a tensor over one chunk) and up as the weights in
    ``OffloadAdafactor.step``, and finally down as the new weights. During
    backward the weights and the finished gradient are both on the GPU already.
    A post-accumulate-grad hook on the gathered ("unsharded") parameter runs
    before FSDP2 copies the gradient down. It hands both to
    ``optimizer.step_param_fused`` and clears the gradient, so ``post_backward``
    finds no gradient for that parameter and skips its reduce and copy. What
    crosses is then the two gathers up and one write-back down.

    Gradient accumulation must be 1: the update happens at each backward.
    The hook calls ``optimizer.step_param_fused`` with the unsharded weights
    when their dtype matches the host shard (at world size 1 they are
    bit-identical, a bf16 copy of a bf16 shard); otherwise it passes none and
    the update reads the host weights, as the 3-pass path does.
    """

    def __init__(self, optimizer: OffloadAdafactor) -> None:
        self.optimizer = optimizer
        self.handles: list[Any] = []
        self.attached = 0

    def attach(self, unsharded: torch.nn.Parameter, owner: torch.nn.Parameter) -> None:
        """Hook ``unsharded`` (the on-device copy) so it steps ``owner`` (the optimizer's parameter)."""
        if not unsharded.requires_grad:
            return
        optimizer = self.optimizer

        def hook(u: torch.nn.Parameter) -> None:
            if u.grad is None:
                return
            with torch.no_grad():
                same_dtype = u.dtype == _local(owner).dtype
                optimizer.step_param_fused(owner, u.grad, u.detach() if same_dtype else None)
            u.grad = None

        self.handles.append(unsharded.register_post_accumulate_grad_hook(hook))
        self.attached += 1

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _fsdp_units(model: Any) -> list[tuple[Any, Any]]:
    """(module, FSDP2 parameter group) for every ``fully_shard`` unit of ``model``."""
    units = []
    for module in model.modules():
        get_state = getattr(module, "_get_fsdp_state", None)
        group = getattr(get_state(), "_fsdp_param_group", None) if callable(get_state) else None
        if group is not None:
            units.append((module, group))
    return units


def install_fused_backward_step(model: Any, optimizer: OffloadAdafactor) -> FusedBackwardStep:
    """Attach :class:`FusedBackwardStep` hooks to a model sharded by ``shard_for_cpu_offload``.

    FSDP2 creates a parameter's unsharded copy at its first gather, so the
    hooks are attached from a one-shot forward pre-hook that runs after
    FSDP2's own (which gathers). The unsharded ``nn.Parameter`` is created
    once and reused every step, so the hook stays on it.

    The only private names used are ``_get_fsdp_state()._fsdp_param_group``
    and its ``fsdp_params[i].sharded_param`` / ``.unsharded_param``. A
    parameter that does not get a hook is not lost: ``optimizer.step()`` steps
    every parameter that still has a host gradient.
    """
    fused = FusedBackwardStep(optimizer)
    wanted = {id(q) for g in optimizer.param_groups for q in g["params"]}
    for module, group in _fsdp_units(model):
        once = _AttachOnFirstForward(fused, group, wanted)
        once.handle = module.register_forward_pre_hook(once, with_kwargs=True)
        fused.handles.append(once.handle)
    return fused


class _AttachOnFirstForward:
    """Forward pre-hook for one FSDP2 unit: attach the fused hooks once, then remove itself."""

    def __init__(self, fused: FusedBackwardStep, group: Any, wanted: set[int]) -> None:
        self.fused = fused
        self.group = group
        self.wanted = wanted
        self.handle: Any = None

    def __call__(self, _module: Any, _args: Any, _kwargs: Any) -> None:
        for fp in self.group.fsdp_params:
            if id(fp.sharded_param) in self.wanted and hasattr(fp, "_unsharded_param"):
                self.fused.attach(fp.unsharded_param, fp.sharded_param)
        self.handle.remove()


# =============================================================================
# Measured fit check (host RAM + VRAM) for full_ft_offload
# =============================================================================
# Every constant below cites the receipt it came from. Receipts are versioned in
# docs/receipts/2026-09-30-offload/ (index in its README.md). All runs are on a
# RunPod RTX 5090 (31.36 GiB), torch 2.8.0+cu128, transformers 5.17.0, seq 512,
# batch 1, BACKPROPAGATE_OFFLOAD_PIN=register (the default), unless noted.
#
# Host RAM: peak RSS (GiB) vs params.
#   q15b_offload_reg.json      Qwen2.5-1.5B            1.544B  train 8.13   with save+reload 10.89  (d291aa2)
#   smollm3_offload_reg.json   SmolLM3-3B              3.075B  train 13.81  with save+reload 15.33  (d291aa2)
#   qwen3_4b_offload_reg.json  Qwen3-4B-Instruct-2507  4.022B  train 19.16  with save+reload 25.07  (d291aa2)
#   q7b_final_receipt.json     Qwen2.5-7B-Instruct     7.616B  train 30.78  with save+reload 32.18  (fc79bb9)
# The least-squares slope of the training peak over those four points is 3.728
# GiB per billion params (4.00 bytes/param: bf16 params + bf16 grads).
_HOST_GIB_PER_BILLION_PARAMS = 3.728
# The fixed term is the UPPER envelope, over the same four runs, of
# (peak with save+reload) - slope * params. Qwen3-4B sets it at 10.07 GiB. It
# covers CUDA and library overhead plus the save -> reload transient; rounded up.
_HOST_FIXED_GIB = 10.1
# VRAM: activations per token (seq x batch). Measured slopes:
#   Qwen2.5-7B: q7b_final_receipt.json (5.25 GiB @ 512 tokens) ->
#   q7b_seq2048.json (6.99 GiB @ 2048) = 1.16 MiB/token.
#   SmolLM3-3B: smollm3_offload_reg.json (2.25 GiB @ 512) -> quality.jsonl
#   offload row (4.28 GiB @ batch 4 x 512) = 1.35 MiB/token.
# The larger slope, rounded up:
_VRAM_MIB_PER_TOKEN = 1.40
# Structural part: 2x the root unit (bf16 embedding + untied LM head, i.e.
# unsharded params plus their grads) + 2 decoder layers (current + prefetch,
# bf16). Structure + activations undershoots the measured allocated peak by at
# most 0.74 GiB, at Qwen3-4B (qwen3_4b_offload_reg.json). Every anchor is
# re-checked in tests/test_offload_fit.py. Margin, rounded up:
_VRAM_MARGIN_GIB = 0.8
# Optimizer-step working set: at most _MAX_CHUNK_NUMEL fp32 elements x ~6 live
# tensors (g, u, w, old, rounded, cached g). Derived from the code, not a
# receipt: 1.5 GiB.
_VRAM_OPTIMIZER_WORKSET_GIB = 6 * 4 * _MAX_CHUNK_NUMEL / 2**30


def offload_host_ram_required_gib(params: float) -> float:
    """Peak host RSS (GiB) needed to train AND save/reload ``params`` with the engine."""
    return _HOST_GIB_PER_BILLION_PARAMS * params / 1e9 + _HOST_FIXED_GIB


def offload_param_ceiling_billions(host_available_gib: float) -> float:
    """The largest model (billions of params) the measured host-RAM model admits."""
    return max(0.0, (host_available_gib - _HOST_FIXED_GIB) / _HOST_GIB_PER_BILLION_PARAMS)


def offload_vram_required_gib(
    root_unit_bytes: float, layer_bytes: float, tokens: int, *, fused: bool = False
) -> float:
    """Peak VRAM (GiB) for one step: max(fwd/bwd working set, optimizer chunk) + margin.

    With the fused backward step the optimizer chunk is live inside backward, on
    top of the forward/backward working set, so the two add. That bound is not
    measured; the max() form is the one the receipts anchor.
    """
    fwd_bwd = (2 * root_unit_bytes + 2 * layer_bytes) / 2**30 + tokens * _VRAM_MIB_PER_TOKEN / 1024
    peak = fwd_bwd + _VRAM_OPTIMIZER_WORKSET_GIB if fused else max(fwd_bwd, _VRAM_OPTIMIZER_WORKSET_GIB)
    return peak + _VRAM_MARGIN_GIB


def model_offload_shape(model: Any) -> tuple[int, int, int]:
    """(params, root-unit bf16 bytes, largest decoder-layer bf16 bytes) of a model.

    Works on a meta-device model (no memory), a loaded model, or a sharded one.
    The root unit is the input embedding, plus the output head when the head is
    not tied to the embedding.
    """
    params = sum(_local(p).numel() for p in model.parameters())
    root = 0
    emb = getattr(model, "get_input_embeddings", lambda: None)()
    head = getattr(model, "get_output_embeddings", lambda: None)()
    emb_w = getattr(emb, "weight", None)
    head_w = getattr(head, "weight", None)
    if emb_w is not None:
        root += emb_w.numel() * 2
    if head_w is not None and head_w is not emb_w:
        root += head_w.numel() * 2
    layer = max(
        (sum(p.numel() for p in lyr.parameters()) * 2 for lyr in _decoder_layers(model)),
        default=0,
    )
    return params, root, layer


def detect_host_ram_gib() -> tuple[float | None, float | None]:
    """(MemTotal, MemAvailable) in GiB, or (None, None).

    Uses psutil when installed (an optional extra), else /proc/meminfo. The
    offload path is Linux / WSL2 only, and /proc/meminfo always exists there.
    Inside WSL2, MemTotal is the VM cap from .wslconfig, which is the budget
    that matters.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except Exception:  # noqa: BLE001, S110 — optional dep; fall through to /proc
        pass  # nosec B110
    try:
        info: dict[str, float] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = int(rest.split()[0]) / 2**20  # kB -> GiB
        return info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        return None, None


def check_offload_fit(
    *,
    params: float,
    root_unit_bytes: float | None,
    layer_bytes: float | None,
    tokens: int,
    host_total_gib: float | None,
    host_available_gib: float | None,
    vram_total_gib: float | None,
    fused: bool = False,
) -> dict[str, Any]:
    """Measured fit check for ``full_ft_offload``; returns a report dict.

    ``fits`` is False when a side is known and too small. An unknown side
    (None) is not judged: the report shows None and the run proceeds.
    """
    ram_need = offload_host_ram_required_gib(params)
    report: dict[str, Any] = {
        "params_billions": round(params / 1e9, 3),
        "host_ram_required_gib": round(ram_need, 1),
        "host_ram_available_gib": None if host_available_gib is None else round(host_available_gib, 1),
        "host_ram_total_gib": None if host_total_gib is None else round(host_total_gib, 1),
        "vram_required_gib": None,
        "vram_total_gib": None if vram_total_gib is None else round(vram_total_gib, 1),
        "tokens_per_step": tokens,
    }
    ram_ok = host_available_gib is None or ram_need <= host_available_gib
    vram_ok = True
    if root_unit_bytes is not None and layer_bytes is not None:
        vram_need = offload_vram_required_gib(root_unit_bytes, layer_bytes, tokens, fused=fused)
        report["vram_required_gib"] = round(vram_need, 1)
        vram_ok = vram_total_gib is None or vram_need <= vram_total_gib
    report["fits_host_ram"] = ram_ok
    report["fits_vram"] = vram_ok
    report["fits"] = ram_ok and vram_ok
    return report


def _decoder_layers(model: Any) -> list[Any]:
    """The repeated transformer blocks to shard one-by-one (HF convention)."""
    names = set(getattr(model, "_no_split_modules", None) or [])
    best: list[Any] = []
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > len(best):
            if not names or type(module[0]).__name__ in names:
                best = list(module)
    return best


def _pin_mode() -> str:
    """How host params are page-locked: "register" (default), "pinned" or "none".

    * ``pinned``: ``CPUOffloadPolicy(pin_memory=True)``. Fastest, but it copies
      params AND every step's grads through torch's pinned caching allocator,
      which rounds each block up to a power of two. For Qwen2.5-7B that grows
      host RAM ~1.8x (a 3584x18944 MLP weight, 136 MB, takes a 256 MB block).
      MEASURED: the 7B run crossed a 60 GB ceiling (~56 GB during steps).
    * ``register`` (default): ``pin_memory=False``, then ``cudaHostRegister``
      the existing param storage in place, with no copy and no rounding. H2D
      stays DMA-fast. Grads use pageable memory: exact size, but a slower
      blocking D2H in backward.
    * ``none``: pageable everything (slowest, smallest).
    """
    mode = os.environ.get("BACKPROPAGATE_OFFLOAD_PIN", "register").strip().lower()
    return mode if mode in {"register", "pinned", "none"} else "register"


def register_host_params(model: Any) -> list[int]:
    """Page-lock each param's CPU storage in place; returns the registered pointers.

    A storage that cudaHostRegister refuses (overlapping pages from an earlier
    registration, a driver limit) stays pageable. That is still correct, just
    a slower H2D copy for that tensor. The count is logged at INFO so a silent
    slowdown can be traced.
    """
    cudart = torch.cuda.cudart()
    done: list[int] = []
    seen: set[int] = set()
    failed = 0
    failed_bytes = 0
    for p in model.parameters():
        st = _local(p).untyped_storage()
        ptr = st.data_ptr()
        if ptr in seen or st.device.type != "cpu" or st.nbytes() == 0:
            continue
        seen.add(ptr)
        if int(cudart.cudaHostRegister(ptr, st.nbytes(), 0)) == 0:
            done.append(ptr)
        else:
            failed += 1
            failed_bytes += st.nbytes()
    logger.info(
        "full_ft_offload: page-locked %d param storages; %d could not be registered "
        "and stay pageable (%.2f GiB, slower H2D for those tensors).",
        len(done), failed, failed_bytes / 2**30,
    )
    return done


def unregister_host_params(ptrs: list[int]) -> None:
    cudart = torch.cuda.cudart()
    for ptr in ptrs:
        cudart.cudaHostUnregister(ptr)


def shard_for_cpu_offload(model: Any, compute_dtype: torch.dtype = torch.bfloat16, mesh: Any = None) -> Any:
    """Apply FSDP2 ``fully_shard`` + CPU offload in place, keeping param dtype.

    Enables activation checkpointing (non-reentrant) first. Returns ``model``,
    which is now an ``FSDPModule`` whose sharded params live on the CPU. ``mesh`` is
    FSDP2's device mesh; None takes the default (CUDA), and tests pass a CPU mesh.
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    param_dtype = next(model.parameters()).dtype
    mp = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=param_dtype)
    off = CPUOffloadPolicy(pin_memory=_pin_mode() == "pinned")
    layers = _decoder_layers(model)
    if not layers:
        raise RuntimeError("full_ft_offload: could not find the model's decoder-layer ModuleList to shard.")
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, offload_policy=off)
    fully_shard(model, mesh=mesh, mp_policy=mp, offload_policy=off)
    return model


def _encode(dataset: Any, tokenizer: Any, max_seq_length: int) -> list[list[int]]:
    cols = set(getattr(dataset, "column_names", []) or [])
    rows: list[list[int]] = []
    for row in dataset:
        if "text" in cols:
            text = row["text"]
        elif "messages" in cols:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        else:
            raise ValueError(
                "full_ft_offload trains on a 'text' or 'messages' column; the dataset has "
                f"{sorted(cols)}."
            )
        ids = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=False)["input_ids"]
        if len(ids) >= 2:
            rows.append(ids)
    if not rows:
        raise ValueError("full_ft_offload: the dataset produced no trainable rows.")
    return rows


def _no_leg(name: str, nbytes: int = 0) -> Any:  # noqa: ARG001 — same signature as LegTrace.leg
    return _NO_LEG


def _sync(device: torch.device) -> None:
    """Wait for the device's queued work (a no-op off CUDA, so the loop runs in CPU tests)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _lr_factor(step: int, total: int, warmup: int, kind: str) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if kind == "constant" or total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    if kind == "linear":
        return max(0.0, 1.0 - progress)
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def run_offload_training(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    max_seq_length: int,
    warmup_steps: int = 0,
    lr_scheduler_type: str = "constant",
    weight_decay: float = 0.0,
    seed: int = 42,
    on_step: Callable[[int, float], None] | None = None,
    fused: bool | None = None,
) -> dict[str, Any]:
    """Shard ``model``, train ``steps`` optimizer steps, and return losses + timing.

    ``fused`` steps each parameter inside backward (:class:`FusedBackwardStep`);
    None reads ``BACKPROPAGATE_OFFLOAD_FUSED``. It needs ``gradient_accumulation == 1``
    and falls back to the 3-pass step otherwise.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)  # nosec B311 — seeded training-data shuffle, not crypto
    device = torch.device("cuda", torch.cuda.current_device())
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = shard_for_cpu_offload(model, compute_dtype=compute_dtype)
    registered = register_host_params(model) if _pin_mode() == "register" else []
    try:
        return _train_loop(
            model, tokenizer, dataset, steps=steps, batch_size=batch_size,
            gradient_accumulation=gradient_accumulation, learning_rate=learning_rate,
            max_seq_length=max_seq_length, warmup_steps=warmup_steps,
            lr_scheduler_type=lr_scheduler_type, weight_decay=weight_decay,
            rng=rng, device=device, on_step=on_step, seed=seed,
            fused=fused_backward_requested() if fused is None else fused,
        )
    finally:
        # Unregister before anything can free the storage (save() only reads it).
        unregister_host_params(registered)


def _train_loop(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    *,
    steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    max_seq_length: int,
    warmup_steps: int,
    lr_scheduler_type: str,
    weight_decay: float,
    rng: random.Random,
    device: torch.device,
    on_step: Callable[[int, float], None] | None,
    seed: int = 0,
    fused: bool = False,
) -> dict[str, Any]:
    optimizer = OffloadAdafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
        # Diagnostic only: BACKPROPAGATE_OFFLOAD_ROUNDING=nearest reproduces the
        # failure mode (bf16 round-to-nearest drops sub-ulp updates) so the gate
        # in scripts/pod_offload_7b.sh can be checked against it on real models.
        stochastic_rounding=os.environ.get("BACKPROPAGATE_OFFLOAD_ROUNDING", "stochastic").strip().lower()
        != "nearest",
    )
    rows = _encode(dataset, tokenizer, max_seq_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    order: list[int] = []

    def next_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal order
        batch = []
        for _ in range(batch_size):
            if not order:
                order = list(range(len(rows)))
                rng.shuffle(order)
            batch.append(rows[order.pop()])
        width = max(len(r) for r in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, r in enumerate(batch):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        labels = ids.masked_fill(mask == 0, -100)
        return ids.to(device), mask.to(device), labels.to(device)

    use_fused = fused
    if fused and gradient_accumulation != 1:
        logger.info(
            "full_ft_offload: the fused backward step needs gradient_accumulation == 1 (got %d); "
            "using the 3-pass optimizer step.",
            gradient_accumulation,
        )
        use_fused = False
    fused_hooks = install_fused_backward_step(model, optimizer) if use_fused else None
    n_trainable = sum(len(g["params"]) for g in optimizer.param_groups)

    mode = trace_mode()
    trace = LegTrace(device) if mode != "off" else None
    undo_probes = install_fsdp_probes(trace) if trace is not None else None
    optimizer.trace = trace
    leg = trace.leg if trace is not None else _no_leg
    profile_at = min(1, steps - 1)  # the second step: lazy init stays out of the profile

    model.train()
    losses: list[float] = []
    step_times: list[float] = []
    retention: list[float] = []
    fused_params: list[int] = []
    samples = 0
    t_start = time.perf_counter()
    try:
        for step in range(steps):
            t0 = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * _lr_factor(step, steps, warmup_steps, lr_scheduler_type)
            total = 0.0
            profiling = trace is not None and mode == "profile" and step == profile_at
            with profile_step(trace) if profiling and trace is not None else _NO_LEG:
                for _ in range(gradient_accumulation):
                    ids, mask, labels = next_batch()
                    with leg("forward"):
                        out = model(input_ids=ids, attention_mask=mask, labels=labels)
                    with leg("backward"):
                        (out.loss / gradient_accumulation).backward()
                    total += float(out.loss.detach()) / gradient_accumulation
                    samples += ids.shape[0]
                with leg("optimizer_step"):
                    optimizer.step()
                retention.append(round(optimizer.last_update_retention or 0.0, 4))
                fused_params.append(optimizer.last_fused_params)
                optimizer.zero_grad(set_to_none=True)
                _sync(device)
            step_times.append(time.perf_counter() - t0)
            losses.append(total)
            if trace is not None:
                trace.end_step(step, step_times[-1] * 1e3)
                if profiling and trace.profile is not None:
                    trace.profile["step"] = step
            logger.info("full_ft_offload step %d/%d loss=%.4f (%.2fs)", step + 1, steps, total, step_times[-1])
            if fused_hooks is not None and step == 0:
                (logger.info if fused_params[0] else logger.warning)(
                    "full_ft_offload fused backward step: %d of %d parameters stepped in backward; "
                    "the rest took the 3-pass step.",
                    fused_params[0], n_trainable,
                )
            if on_step is not None:
                try:
                    on_step(step + 1, total)
                except Exception as cb_err:  # noqa: BLE001 — callback isolation contract
                    logger.warning("on_step callback raised: %s", cb_err)
    finally:
        if fused_hooks is not None:
            fused_hooks.remove()
        if undo_probes is not None:
            undo_probes()
        optimizer.trace = None
    result: dict[str, Any] = {
        "model": model,
        "losses": losses,
        "step_times": step_times,
        "update_retention": retention,
        "samples_seen": samples,
        "duration_seconds": time.perf_counter() - t_start,
        "optimizer": optimizer,
        "fused": use_fused,
        "fused_params": fused_params,
    }
    if trace is not None:
        result["trace"] = trace.summary({"pin": _pin_mode(), "trace_mode": mode, "fused": use_fused})
    return result
