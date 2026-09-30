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
  rounding keeps every update in expectation at zero extra bytes.

Trade-offs that callers and docs must state:
* Adafactor, not AdamW (no momentum; the second moment is factored).
* There are no fp32 master weights. Precision comes from stochastic rounding.
* The loop is plain causal-LM SFT over the whole sequence. It has no packing,
  no response-only masking, and no intermediate checkpoints (save() at the end).
* It needs Linux / WSL2 with NCCL (FSDP2); Windows-native fails fast with
  DEP_FSDP_UNAVAILABLE.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Rows of a 2-D parameter processed per optimizer chunk on the GPU. Bounds the
# transient fp32 working set of the step (a 7B embedding is 545M elements).
_MAX_CHUNK_NUMEL = 1 << 26


def _local(t: Any) -> Any:
    """The local shard of a DTensor, or the tensor itself."""
    to_local = getattr(t, "to_local", None)
    return to_local() if callable(to_local) else t


def stochastic_round_to_bf16_(dst: torch.Tensor, src_fp32: torch.Tensor) -> None:
    """Write fp32 ``src_fp32`` into bf16 ``dst`` with stochastic rounding.

    Adds uniform noise below bf16's lowest kept bit, then truncates. The
    expected value of the result equals the fp32 input, so tiny updates survive
    on average instead of rounding to zero.
    """
    bits = src_fp32.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=bits.device)
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

    def _write_back(self, host: torch.Tensor, new_fp32: torch.Tensor) -> None:
        if host.dtype == torch.bfloat16 and self.stochastic_rounding:
            tmp = torch.empty_like(new_fp32, dtype=torch.bfloat16)
            stochastic_round_to_bf16_(tmp, new_fp32)
            host.copy_(tmp)
        else:
            host.copy_(new_fp32)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:  # type: ignore[override]
        loss = closure() if closure is not None else None
        dev = self.device
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                w_host = _local(p)
                g_host = _local(p.grad)
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

                if w_host.dim() < 2:
                    g = g_host.to(dev, non_blocking=True).float()
                    st["v"].mul_(beta2).add_(g * g + self.eps, alpha=1.0 - beta2)
                    u = g / st["v"].sqrt()
                    u.div_(max(1.0, float(u.pow(2).mean().sqrt()) / self.clip_threshold))
                    w = w_host.to(dev, non_blocking=True).float()
                    if wd:
                        w.mul_(1.0 - lr * wd)
                    w.sub_(u, alpha=lr)
                    self._write_back(w_host, w)
                    continue

                rows = w_host.shape[0]
                g2d_host = g_host.reshape(rows, -1)
                w2d_host = w_host.view(rows, -1)  # a VIEW: writes must land in the param
                cols = g2d_host.shape[1]
                chunk = max(1, _MAX_CHUNK_NUMEL // max(1, cols))
                spans = [(i, min(rows, i + chunk)) for i in range(0, rows, chunk)]
                cached: dict[str, torch.Tensor] = {}  # single-chunk params keep g on the device

                def _g(
                    a: int,
                    b: int,
                    src: torch.Tensor = g2d_host,
                    single: bool = len(spans) == 1,
                    cache: dict[str, torch.Tensor] = cached,
                ) -> torch.Tensor:
                    if single:
                        if "g" not in cache:
                            cache["g"] = src[a:b].to(dev, non_blocking=True).float()
                        return cache["g"]
                    return src[a:b].to(dev, non_blocking=True).float()

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

                # Pass 2: RMS of the update, for clipping.
                sumsq = torch.zeros((), dtype=torch.float32, device=dev)
                for a, b in spans:
                    u = _g(a, b) * row_norm[a:b].rsqrt().unsqueeze(1) * col_rsqrt
                    sumsq.add_(u.pow(2).sum())
                rms = float((sumsq / (rows * cols)).sqrt())
                scale = lr / max(1.0, rms / self.clip_threshold)

                # Pass 3: apply and write back.
                for a, b in spans:
                    u = _g(a, b) * row_norm[a:b].rsqrt().unsqueeze(1) * col_rsqrt
                    w = w2d_host[a:b].to(dev, non_blocking=True).float()
                    if wd:
                        w.mul_(1.0 - lr * wd)
                    w.sub_(u, alpha=scale)
                    self._write_back(w2d_host[a:b], w)
        return loss


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
    """Page-lock each param's CPU storage in place; returns the registered pointers."""
    cudart = torch.cuda.cudart()
    done: list[int] = []
    for p in model.parameters():
        st = _local(p).untyped_storage()
        ptr = st.data_ptr()
        if ptr in done or st.device.type != "cpu" or st.nbytes() == 0:
            continue
        if int(cudart.cudaHostRegister(ptr, st.nbytes(), 0)) == 0:
            done.append(ptr)
    return done


def unregister_host_params(ptrs: list[int]) -> None:
    cudart = torch.cuda.cudart()
    for ptr in ptrs:
        cudart.cudaHostUnregister(ptr)


def shard_for_cpu_offload(model: Any, compute_dtype: torch.dtype = torch.bfloat16) -> Any:
    """Apply FSDP2 ``fully_shard`` + CPU offload in place, keeping param dtype.

    Enables activation checkpointing (non-reentrant) first. Returns ``model``,
    which is now an ``FSDPModule`` whose sharded params live on the CPU.
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
        fully_shard(layer, mp_policy=mp, offload_policy=off)
    fully_shard(model, mp_policy=mp, offload_policy=off)
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
) -> dict[str, Any]:
    """Shard ``model``, train ``steps`` optimizer steps, and return losses + timing."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
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
            rng=rng, device=device, on_step=on_step,
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
) -> dict[str, Any]:
    optimizer = OffloadAdafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
        device=device,
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

    model.train()
    losses: list[float] = []
    step_times: list[float] = []
    samples = 0
    t_start = time.perf_counter()
    for step in range(steps):
        t0 = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * _lr_factor(step, steps, warmup_steps, lr_scheduler_type)
        total = 0.0
        for _ in range(gradient_accumulation):
            ids, mask, labels = next_batch()
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            (out.loss / gradient_accumulation).backward()
            total += float(out.loss.detach()) / gradient_accumulation
            samples += ids.shape[0]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)
        losses.append(total)
        logger.info("full_ft_offload step %d/%d loss=%.4f (%.2fs)", step + 1, steps, total, step_times[-1])
        if on_step is not None:
            try:
                on_step(step + 1, total)
            except Exception as cb_err:  # noqa: BLE001 — callback isolation contract
                logger.warning("on_step callback raised: %s", cb_err)
    return {
        "model": model,
        "losses": losses,
        "step_times": step_times,
        "samples_seen": samples,
        "duration_seconds": time.perf_counter() - t_start,
        "optimizer": optimizer,
    }
