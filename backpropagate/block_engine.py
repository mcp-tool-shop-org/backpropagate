"""Engine B — block-coordinate AdamW for full fine-tuning (phase 1).

Full-parameter fine-tuning where only ONE block of parameters trains at a time
(Luo, Yu, Li 2024, *BAdam: A Memory Efficient Full Parameter Optimization
Method for Large Language Models*, NeurIPS; arXiv:2404.02827). The whole model
stays on the GPU in its 16-bit storage dtype. For ``K`` optimizer steps one
block — a transformer layer, the input embeddings, or the output head — has
``requires_grad=True`` and is updated with AdamW; every other parameter is
frozen. Then the block is written back to 16-bit, its optimizer state is
dropped, and the next block becomes active.

This module is our own implementation from the paper and the design doc
(``docs/full-ft-engines-design-2026-09-30.md``, PR #232). BAdam's reference
code (github.com/Ledzy/BAdam, Apache-2.0) was read to confirm behaviour
(default order, the default treatment of embeddings/head); no code was copied.

Design decisions (each one is tested in ``tests/test_block_engine.py``)
----------------------------------------------------------------------
* **It is a ``torch.optim.Optimizer``, not a training loop.** It is handed to
  TRL's ``SFTTrainer`` through ``optimizers=(opt, None)``, so packing,
  checkpoints, resume, gradient accumulation and global grad-norm clipping
  are the stock HF Trainer machinery. Steps are counted in *optimizer* steps,
  so ``K`` composes with gradient accumulation.
* **Precision.** While a block is active its parameters are held in fp32
  *in the module itself* (``param.data`` is upcast in place); the forward pass
  runs under the trainer's bf16 autocast, which casts the fp32 weights to bf16
  for the matmuls. AdamW state is fp32. On deactivation the fp32 values are
  written back to the storage dtype once per visit — with **stochastic
  rounding** by default (unbiased: random bits are added below the bf16
  mantissa before truncation), or round-to-nearest for the A/B.
* **Backward scope.** Frozen parameters have ``requires_grad=False``, so
  autograd never allocates their gradients. transformers >= 4.35 installs an
  "input requires grad" hook on the embeddings when gradient checkpointing is
  enabled; that hook would make backward traverse (and recompute) every layer
  below the active block. The engine removes it at train begin, so backward
  stops at the active block.
* **Resume.** ``state_dict()`` carries the schedule (active block,
  step-within-block, visit count, order and its RNG state) plus the active
  block's fp32 master and AdamW state, keyed by parameter name, so a TRL
  checkpoint resumes the same schedule. The stochastic-rounding RNG is derived
  from ``(seed, visit)`` rather than carried, so it is reproducible too.

Seam for Engine C (host-to-GPU block swap, not built here): the partition
(:class:`BlockPartition`) and the ``_activate`` / ``_deactivate`` pair of
:class:`BlockCoordinateOptimizer` are the only places that touch a block's
storage. A transfer layer makes a block's weights resident before
``_activate`` and may evict a frozen block after ``_deactivate``; the AdamW
maths, the schedule and the checkpoint format do not change.

This module imports torch at import time; the package imports it lazily (only
when ``full_ft_engine='block'`` is used).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .exceptions import CheckpointError, InvalidSettingError, TrainingError

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCK_ORDERS",
    "BLOCK_WRITEBACK_MODES",
    "DEFAULT_BLOCK_ORDER",
    "DEFAULT_BLOCK_WRITEBACK",
    "DEFAULT_SWITCH_BLOCK_EVERY",
    "FULL_FT_ENGINES",
    "Block",
    "BlockCoordinateOptimizer",
    "BlockEngineFit",
    "BlockPartition",
    "build_block_engine_callback",
    "build_for_sft_trainer",
    "drop_input_require_grads_hook",
    "estimate_block_engine_vram",
    "find_layer_list",
    "fit_from_model",
    "partition_model",
    "round_nearest_to",
    "stochastic_round_to_bf16",
    "validate_block_engine_settings",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
#: Accepted values of ``Trainer(full_ft_engine=...)``. ``"default"`` (or None)
#: is the library's standard pure-GPU full fine-tuning path.
FULL_FT_ENGINES: tuple[str, ...] = ("default", "block")

#: Block orders. ``random`` is random reshuffling: a fresh permutation of all
#: blocks each block-epoch. It is what the paper's Algorithm 1 and its
#: experiments use ("The ordering strategy in the partition π of BAdam is
#: random reshuffling") and what the BAdam README recommends; the paper's
#: ordering ablation (App. C.1) finds all three converge alike, descending a
#: little slower at first. ``ascending`` = input side to output side.
BLOCK_ORDERS: tuple[str, ...] = ("random", "ascending", "descending")
DEFAULT_BLOCK_ORDER = "random"

#: Write-back rounding from the fp32 master to the 16-bit storage dtype.
BLOCK_WRITEBACK_MODES: tuple[str, ...] = ("stochastic", "nearest")
DEFAULT_BLOCK_WRITEBACK = "stochastic"

#: K — optimizer steps per block visit. The design picks 50, the low end of
#: the paper's suggested ``min(max(n/(B·D), 50), 100)``.
DEFAULT_SWITCH_BLOCK_EVERY = 50

#: Elements per chunk when rounding a block back to 16-bit. Bounds the
#: transient int32 noise buffer to 64 MiB whatever the block size.
_ROUND_CHUNK_ELEMENTS = 1 << 24

_STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Validation (shared by Trainer construction and the optimizer)
# ---------------------------------------------------------------------------
def validate_block_engine_settings(
    *,
    switch_block_every: Any,
    block_order: Any,
    block_writeback: Any,
) -> None:
    """Raise ``InvalidSettingError`` (CONFIG_INVALID_SETTING) on a bad knob."""
    if (
        isinstance(switch_block_every, bool)
        or not isinstance(switch_block_every, int)
        or switch_block_every < 1
    ):
        raise InvalidSettingError(
            setting_name="switch_block_every",
            value=switch_block_every,
            expected="a positive integer (optimizer steps per block visit)",
            suggestion=(
                f"Use switch_block_every={DEFAULT_SWITCH_BLOCK_EVERY} (the default). "
                "BAdam suggests min(max(n/(B*D), 50), 100) for n examples, "
                "effective batch B and D blocks."
            ),
        )
    if block_order not in BLOCK_ORDERS:
        raise InvalidSettingError(
            setting_name="block_order",
            value=block_order,
            expected=f"one of {set(BLOCK_ORDERS)}",
            suggestion="Use block_order='random' (random reshuffling, the paper's choice).",
        )
    if block_writeback not in BLOCK_WRITEBACK_MODES:
        raise InvalidSettingError(
            setting_name="block_writeback",
            value=block_writeback,
            expected=f"one of {set(BLOCK_WRITEBACK_MODES)}",
            suggestion="Use block_writeback='stochastic' (the default) or 'nearest'.",
        )


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------
def stochastic_round_to_bf16(
    src: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Round an fp32 tensor to bf16 with unbiased stochastic rounding.

    bf16 is the top 16 bits of an fp32. Adding a uniform random integer in
    ``[0, 2**16)`` to the fp32 bit pattern and then clearing the low 16 bits
    rounds the magnitude up with probability equal to the discarded fraction,
    so ``E[round(x)] == x`` (for finite x away from the bf16 overflow edge).
    The sign bit is untouched, so negative values round symmetrically. Values
    that are already bf16-representable (low 16 bits zero) come back exactly.

    The final ``.to(torch.bfloat16)`` converts a value whose low 16 bits are
    zero, which is exact — the rounding decision is made here, once, and
    nothing downstream re-rounds it.
    """
    if src.dtype != torch.float32:
        raise TypeError(f"stochastic_round_to_bf16 expects float32, got {src.dtype}")
    flat = src.detach().contiguous().reshape(-1)
    out = torch.empty(flat.shape, dtype=torch.bfloat16, device=flat.device)
    for start in range(0, flat.numel(), _ROUND_CHUNK_ELEMENTS):
        chunk = flat[start : start + _ROUND_CHUNK_ELEMENTS]
        noise = torch.randint(
            0, 1 << 16, chunk.shape, dtype=torch.int32,
            device=chunk.device, generator=generator,
        )
        noise.add_(chunk.view(torch.int32))
        noise.bitwise_and_(-65536)  # 0xFFFF0000: keep sign, exponent, top 7 mantissa bits
        out[start : start + chunk.numel()] = noise.view(torch.float32)
    return out.view(src.shape)


def round_nearest_to(src: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round-to-nearest-even cast (PyTorch's default conversion)."""
    return src.detach().to(dtype)


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Block:
    """One block: the parameters that train together for one visit."""

    index: int
    name: str
    param_names: tuple[str, ...]
    numel: int


@dataclass(frozen=True)
class BlockPartition:
    """The model's parameters split into blocks, input side first.

    ``frozen_param_names`` are parameters that belong to no block (the
    embeddings and head when ``include_embeddings=False``); they never train.
    ``tied`` records whether the input and output embeddings share storage —
    in that case the shared weight is in exactly one block (``embed+head``).
    """

    blocks: tuple[Block, ...]
    layer_list_name: str
    frozen_param_names: tuple[str, ...]
    tied: bool
    total_numel: int

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.blocks]

    @property
    def max_block_numel(self) -> int:
        return max(b.numel for b in self.blocks)

    def summary(self) -> dict[str, Any]:
        return {
            "num_blocks": len(self.blocks),
            "layer_list": self.layer_list_name,
            "tied_embeddings": self.tied,
            "max_block": max(self.blocks, key=lambda b: b.numel).name,
            "max_block_params": self.max_block_numel,
            "total_params": self.total_numel,
            "frozen_params": len(self.frozen_param_names),
        }


def _numel(p: Any) -> int:
    return int(p.numel())


def find_layer_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Find the model's repeated transformer-layer list, generically.

    A decoder is a stack of identical blocks held in one ``nn.ModuleList``
    (``model.layers`` in Llama/Qwen, ``transformer.h`` in GPT-2, ...). We look
    for ``ModuleList`` s whose children all share one class — the same
    structural signal accelerate's ``_no_split_modules`` and PEFT's
    ``layers_pattern`` encode by name — and take the one holding the most
    parameters (a per-layer expert list inside an MoE layer is nested *inside*
    the layer list, so it is always smaller). When the model declares
    ``_no_split_modules`` and a candidate's child class is named there, that
    candidate wins.
    """
    candidates: list[tuple[int, str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        if len({type(child) for child in module}) != 1:
            continue
        n = sum(_numel(p) for p in module.parameters())
        if n > 0:
            candidates.append((n, name, module))
    if not candidates:
        raise TrainingError(
            f"full_ft_engine='block' could not find a repeated transformer-layer "
            f"list in {type(model).__name__} (no nn.ModuleList of identical "
            f"modules with parameters).",
            code="RUNTIME_TRAINING_FAILED",
            suggestion=(
                "The block-coordinate engine partitions by transformer layer. "
                "Use the default full fine-tuning engine for this architecture "
                "(drop --full-ft-engine block), or mode='lora'."
            ),
        )
    no_split = set(getattr(model, "_no_split_modules", None) or [])
    preferred = [c for c in candidates if type(c[2][0]).__name__ in no_split]
    pool = preferred or candidates
    _, name, module = max(pool, key=lambda c: c[0])
    return name, module


def partition_model(model: nn.Module, *, include_embeddings: bool = True) -> BlockPartition:
    """Partition ``model``'s parameters into blocks.

    * Each element of the repeated layer list is one block (``layers.<i>``).
    * Parameters registered before the layer list (token + position
      embeddings) form the ``embed`` block; parameters after it (final norm,
      untied output head) form the ``head`` block.
    * Tied input/output embeddings: ``named_parameters()`` yields the shared
      weight once, so it lands in exactly one block. With the head tied there
      is no separate head matrix, and the post-layer parameters (the final
      norm) join the embedding block as ``embed+head`` rather than forming a
      near-empty block that would burn K steps training a norm vector.
    * ``include_embeddings=False`` freezes the embed/head parameters for the
      whole run (BAdam's own default; see the module docstring).

    The input embedding module (``get_input_embeddings()``) is always put on
    the input side even if the model registers it after the layers.
    """
    layer_list_name, layers = find_layer_list(model)
    prefix = f"{layer_list_name}." if layer_list_name else ""

    input_emb_ids: set[int] = set()
    output_emb_ids: set[int] = set()
    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    try:
        emb = get_in() if callable(get_in) else None
        if emb is not None:
            input_emb_ids = {id(p) for p in emb.parameters()}
    except (NotImplementedError, AttributeError):
        input_emb_ids = set()
    try:
        head = get_out() if callable(get_out) else None
        if head is not None:
            output_emb_ids = {id(p) for p in head.parameters()}
    except (NotImplementedError, AttributeError):
        output_emb_ids = set()
    tied = bool(input_emb_ids & output_emb_ids)
    if not tied:
        # Generic fallback: any storage reachable under two names is tied.
        seen: dict[int, str] = {}
        for _name, p in model.named_parameters(remove_duplicate=False):
            if id(p) in seen:
                tied = True
                break
            seen[id(p)] = _name

    per_layer: dict[int, list[str]] = {i: [] for i in range(len(layers))}
    pre: list[str] = []
    post: list[str] = []
    numel: dict[str, int] = {}
    seen_layers = False
    for name, p in model.named_parameters():  # remove_duplicate=True: tied once
        numel[name] = _numel(p)
        if prefix and name.startswith(prefix):
            idx_str = name[len(prefix):].split(".", 1)[0]
            if idx_str.isdigit() and int(idx_str) in per_layer:
                per_layer[int(idx_str)].append(name)
                seen_layers = True
                continue
        if id(p) in input_emb_ids or not seen_layers:
            pre.append(name)
        else:
            post.append(name)

    blocks: list[Block] = []
    frozen: list[str] = []

    def _add(block_name: str, names: Sequence[str]) -> None:
        if names:
            blocks.append(
                Block(
                    index=len(blocks),
                    name=block_name,
                    param_names=tuple(names),
                    numel=sum(numel[n] for n in names),
                )
            )

    if include_embeddings:
        if tied:
            _add("embed+head", pre + post)
        else:
            _add("embed", pre)
    else:
        frozen.extend(pre)
        if tied:
            frozen.extend(post)
    for i in range(len(layers)):
        _add(f"{layer_list_name}.{i}" if layer_list_name else str(i), per_layer[i])
    if not tied:
        if include_embeddings:
            _add("head", post)
        else:
            frozen.extend(post)

    if not blocks:
        raise TrainingError(
            "full_ft_engine='block' found no trainable parameters to partition.",
            code="RUNTIME_TRAINING_FAILED",
        )
    return BlockPartition(
        blocks=tuple(blocks),
        layer_list_name=layer_list_name,
        frozen_param_names=tuple(frozen),
        tied=tied,
        total_numel=sum(numel.values()),
    )


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------
def _is_low_precision(dtype: torch.dtype) -> bool:
    return dtype in (torch.bfloat16, torch.float16)


class BlockCoordinateOptimizer(Optimizer):
    """AdamW over one active block at a time (see the module docstring).

    Keyword arguments: ``lr``, ``betas``, ``eps``, ``weight_decay`` (the AdamW
    hyperparameters, applied to the active block only), ``switch_block_every``
    (K), ``block_order``, ``block_writeback``, ``seed`` (random order and the
    stochastic-rounding stream), ``include_embeddings``, ``partition`` (a
    precomputed :class:`BlockPartition`) and ``upcast`` (hold the active block
    in fp32 while it trains; only meaningful when the model is stored in
    16-bit).

    Two parameter groups exist for the whole run — decay (matrices) and
    no-decay (biases, norm vectors) — whose ``params`` lists are swapped on
    every switch, so an LR scheduler built on this optimizer stays valid.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        switch_block_every: int = DEFAULT_SWITCH_BLOCK_EVERY,
        block_order: str = DEFAULT_BLOCK_ORDER,
        block_writeback: str = DEFAULT_BLOCK_WRITEBACK,
        seed: int = 0,
        include_embeddings: bool = True,
        partition: BlockPartition | None = None,
        upcast: bool = True,
    ) -> None:
        validate_block_engine_settings(
            switch_block_every=switch_block_every,
            block_order=block_order,
            block_writeback=block_writeback,
        )
        self.model = model
        self.partition = partition or partition_model(
            model, include_embeddings=include_embeddings
        )
        self.switch_block_every = int(switch_block_every)
        self.block_order = block_order
        self.block_writeback = block_writeback
        self.seed = int(seed)
        self.upcast = bool(upcast)

        self._params: dict[str, nn.Parameter] = dict(model.named_parameters())
        self._storage_dtype: dict[str, torch.dtype] = {
            n: p.dtype for n, p in self._params.items()
        }
        self._orig_requires_grad: dict[str, bool] = {
            n: bool(p.requires_grad) for n, p in self._params.items()
        }
        # Everything starts frozen; _activate un-freezes one block.
        for p in self._params.values():
            p.requires_grad_(False)

        self._order_gen = torch.Generator().manual_seed(self.seed)
        self._order: list[int] = []
        self._order_pos = 0
        self._active: int | None = None
        self._steps_in_block = 0
        self._global_step = 0
        self._visit = -1  # incremented on every activation
        self._finalized = False
        self.switch_log: list[dict[str, Any]] = []

        defaults = {"lr": lr, "betas": tuple(betas), "eps": eps, "weight_decay": weight_decay}
        super().__init__(
            [
                {"params": [], "weight_decay": weight_decay},
                {"params": [], "weight_decay": 0.0},
            ],
            defaults,
        )
        self._activate(self._next_block_index())

    # -- block bookkeeping -----------------------------------------------------
    @property
    def active_block(self) -> Block | None:
        return None if self._active is None else self.partition.blocks[self._active]

    @property
    def active_block_index(self) -> int | None:
        return self._active

    @property
    def steps_in_block(self) -> int:
        return self._steps_in_block

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def visit(self) -> int:
        return self._visit

    def block_params(self, index: int) -> list[nn.Parameter]:
        return [self._params[n] for n in self.partition.blocks[index].param_names]

    def _split_decay(self, blk: Block) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        # HF Trainer's convention in effect: biases and norm vectors (ndim < 2)
        # get no weight decay; matrices and embeddings do.
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for n in blk.param_names:
            p = self._params[n]
            (decay if p.ndim >= 2 else no_decay).append(p)
        return decay, no_decay

    def _new_order(self) -> list[int]:
        d = len(self.partition.blocks)
        if self.block_order == "ascending":
            return list(range(d))
        if self.block_order == "descending":
            return list(range(d - 1, -1, -1))
        return [int(i) for i in torch.randperm(d, generator=self._order_gen).tolist()]

    def _next_block_index(self) -> int:
        if self._order_pos >= len(self._order):
            self._order = self._new_order()
            self._order_pos = 0
        idx = self._order[self._order_pos]
        self._order_pos += 1
        return idx

    def _sr_generator(self, device: torch.device) -> torch.Generator:
        # Derived from (seed, visit), not carried in the checkpoint: a resumed
        # run rounds exactly like an uninterrupted one.
        g = torch.Generator(device=device)
        g.manual_seed((self.seed * 1_000_003 + self._visit * 7_919 + 17) % (2**63 - 1))
        return g

    # -- activation / deactivation (the Engine C seam) ---------------------------
    def _activate(self, index: int) -> None:
        if self._active is not None:
            raise RuntimeError("block engine: _activate called while a block is active")
        self._visit += 1
        blk = self.partition.blocks[index]
        for n in blk.param_names:
            p = self._params[n]
            if self.upcast and _is_low_precision(p.dtype):
                p.data = p.data.float()
            p.requires_grad_(True)
        decay, no_decay = self._split_decay(blk)
        self.param_groups[0]["params"] = decay
        self.param_groups[1]["params"] = no_decay
        self._active = index
        self._steps_in_block = 0
        logger.debug(
            "block engine: activated block %d (%s, %d params) visit=%d",
            index, blk.name, blk.numel, self._visit,
        )

    def _deactivate(self) -> None:
        if self._active is None:
            return
        blk = self.partition.blocks[self._active]
        params = [self._params[n] for n in blk.param_names]
        # Free optimizer state and grads first, so the write-back's transient
        # buffers reuse that memory rather than adding to the peak.
        for p in params:
            self.state.pop(p, None)
            p.grad = None
            p.requires_grad_(False)
        gen: torch.Generator | None = None
        for n, p in zip(blk.param_names, params):
            target = self._storage_dtype[n]
            if p.dtype == target:
                continue
            if (
                self.block_writeback == "stochastic"
                and target == torch.bfloat16
                and p.dtype == torch.float32
            ):
                if gen is None:
                    gen = self._sr_generator(p.device)
                p.data = stochastic_round_to_bf16(p.data, gen)
            else:
                if self.block_writeback == "stochastic":
                    logger.warning(
                        "block engine: stochastic write-back is implemented for "
                        "bf16 storage only; rounding %s to nearest.", target,
                    )
                p.data = round_nearest_to(p.data, target)
        self.param_groups[0]["params"] = []
        self.param_groups[1]["params"] = []
        self.switch_log.append(
            {
                "block": blk.name,
                "index": self._active,
                "visit": self._visit,
                "steps": self._steps_in_block,
                "at_global_step": self._global_step,
                "writeback": self.block_writeback,
            }
        )
        self._active = None

    def switch(self) -> None:
        """Write the active block back and activate the next one."""
        self._deactivate()
        self._activate(self._next_block_index())

    def finalize(self) -> None:
        """Write the active block back and restore every parameter's original
        ``requires_grad``. Idempotent. Afterwards the model is a plain 16-bit
        model again, safe to save or export."""
        if self._finalized:
            return
        self._deactivate()
        for n, p in self._params.items():
            p.requires_grad_(self._orig_requires_grad[n])
        self._finalized = True

    # -- the update ---------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._finalized:
            raise TrainingError(
                "block engine: step() after finalize().", code="RUNTIME_TRAINING_FAILED"
            )
        # The per-parameter update lives in its own method so its locals (the
        # last parameter's grad, moments and denominator) are released before
        # a switch writes the block back. Measured on the pod: with the update
        # inline in step(), a 7B head/embedding block's write-back ran with
        # ~8.7 GB of those tensors still referenced by this frame.
        any_grad = False
        for group in self.param_groups:
            any_grad = self._update_group(group) or any_grad
        if not any_grad:
            blk = self.active_block
            raise TrainingError(
                f"block engine: no gradient reached the active block "
                f"({blk.name if blk else '?'}) at optimizer step {self._global_step + 1}.",
                code="RUNTIME_TRAINING_FAILED",
                suggestion=(
                    "Reentrant gradient checkpointing drops gradients when a "
                    "checkpointed segment's inputs do not require grad. The "
                    "library configures use_reentrant=False; keep it False if "
                    "you override gradient_checkpointing_kwargs."
                ),
            )
        self._global_step += 1
        self._steps_in_block += 1
        if self._steps_in_block >= self.switch_block_every:
            self.switch()
        return loss

    def _update_group(self, group: dict[str, Any]) -> bool:
        """AdamW on one parameter group; True when any parameter had a grad."""
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        wd = float(group["weight_decay"])
        seen = False
        for p in group["params"]:
            if p.grad is None:
                continue
            seen = True
            st = self.state[p]
            if not st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            st["step"] += 1
            t = st["step"]
            g = p.grad.float()
            exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]
            # Decoupled weight decay, then Adam — torch.optim.AdamW's maths.
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            exp_avg.lerp_(g, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - beta2**t)).add_(eps)
            step_size = lr / (1.0 - beta1**t)
            if p.dtype == torch.float32:
                p.addcdiv_(exp_avg, denom, value=-step_size)
            else:  # upcast=False on a 16-bit model: update in storage dtype
                p.sub_(exp_avg.div(denom).mul_(step_size).to(p.dtype))
        return seen

    # -- checkpoint state -----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        blk = self.active_block
        master: dict[str, torch.Tensor] = {}
        adam: dict[str, dict[str, Any]] = {}
        if blk is not None:
            for n in blk.param_names:
                p = self._params[n]
                master[n] = p.data  # a reference, as torch's own state_dict does
                st = self.state.get(p)
                if st:
                    adam[n] = {
                        "step": int(st["step"]),
                        "exp_avg": st["exp_avg"],
                        "exp_avg_sq": st["exp_avg_sq"],
                    }
        groups = [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != "params"}
            for g in self.param_groups
        ]
        return {
            "state": {},
            "param_groups": groups,
            "block_engine": {
                "version": _STATE_VERSION,
                "block_names": self.partition.names,
                "switch_block_every": self.switch_block_every,
                "block_order": self.block_order,
                "block_writeback": self.block_writeback,
                "seed": self.seed,
                "active_block": -1 if self._active is None else int(self._active),
                "steps_in_block": int(self._steps_in_block),
                "global_step": int(self._global_step),
                "visit": int(self._visit),
                "order": [int(i) for i in self._order],
                "order_pos": int(self._order_pos),
                "order_rng_state": self._order_gen.get_state(),
                "master": master,
                "adam": adam,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        be = state_dict.get("block_engine") if isinstance(state_dict, dict) else None
        if not isinstance(be, dict):
            raise CheckpointError(
                "load", "optimizer.pt",
                "the optimizer state was not written by the block-coordinate engine "
                "(no 'block_engine' section); resume with the engine the checkpoint "
                "was trained with.",
            )
        saved_names = list(be.get("block_names", []))
        if saved_names != self.partition.names:
            raise CheckpointError(
                "load", "optimizer.pt",
                f"the checkpoint's block partition ({len(saved_names)} blocks) does "
                f"not match this model's ({len(self.partition.names)} blocks): a "
                "different model or a different block_train_embeddings setting.",
            )
        for key in ("switch_block_every", "block_order", "block_writeback", "seed"):
            if be.get(key) != getattr(self, key):
                logger.warning(
                    "block engine: checkpoint %s=%r differs from this run's %r; the "
                    "checkpoint's schedule position is restored and this run's "
                    "setting applies from here on.",
                    key, be.get(key), getattr(self, key),
                )
        for group, saved in zip(self.param_groups, state_dict.get("param_groups", [])):
            for k, v in saved.items():
                if k != "params":
                    group[k] = tuple(v) if k == "betas" else v

        target = int(be["active_block"])
        current = -1 if self._active is None else self._active
        if target != current:
            # At resume the currently active block holds the values the model
            # checkpoint just loaded — all 16-bit representable — so writing it
            # back is exact under either rounding mode.
            self._deactivate()
        self._order = [int(i) for i in be["order"]]
        self._order_pos = int(be["order_pos"])
        self._order_gen.set_state(be["order_rng_state"].cpu())
        self.state.clear()
        if target >= 0:
            if self._active is None:
                self._visit = int(be["visit"]) - 1
                self._activate(target)
            self._visit = int(be["visit"])
            self._steps_in_block = int(be["steps_in_block"])
            for n in self.partition.blocks[target].param_names:
                p = self._params[n]
                src = be["master"].get(n)
                if src is not None and src.data_ptr() != p.data.data_ptr():
                    p.data.copy_(src.to(device=p.device, dtype=p.dtype))
                st = be["adam"].get(n)
                if st:
                    self.state[p] = {
                        "step": int(st["step"]),
                        "exp_avg": st["exp_avg"].to(device=p.device, dtype=torch.float32),
                        "exp_avg_sq": st["exp_avg_sq"].to(device=p.device, dtype=torch.float32),
                    }
        self._global_step = int(be["global_step"])
        self._finalized = False

    def engine_summary(self) -> dict[str, Any]:
        """A JSON-safe description of the partition and what happened."""
        out = dict(self.partition.summary())
        out.update(
            {
                "engine": "block",
                "switch_block_every": self.switch_block_every,
                "block_order": self.block_order,
                "block_writeback": self.block_writeback,
                "upcast_active_block_fp32": self.upcast,
                "optimizer_steps": self._global_step,
                "block_visits": self._visit + 1,
                "switches": list(self.switch_log),
            }
        )
        return out


# ---------------------------------------------------------------------------
# HF Trainer callback
# ---------------------------------------------------------------------------
def drop_input_require_grads_hook(model: Any) -> bool:
    """Remove transformers' "input embeddings require grad" hook if present.

    ``PreTrainedModel.gradient_checkpointing_enable`` (transformers >= 4.35)
    registers a forward hook that marks the embedding output as requiring
    grad, for PEFT's benefit. Under block-coordinate training it would make
    backward walk — and, with checkpointing, recompute — every layer below the
    active block to produce activation gradients nobody uses. Returns True when
    a hook was removed.
    """
    if getattr(model, "_require_grads_hook", None) is None:
        return False
    disable = getattr(model, "disable_input_require_grads", None)
    if callable(disable):
        disable()
        return True
    return False


def build_block_engine_callback(optimizer: Any) -> Any:
    """A ``TrainerCallback`` that keeps the engine consistent with the loop:
    drops the input-grad hook once gradient checkpointing is on (train begin)
    and writes the active block back at train end."""
    from transformers import TrainerCallback

    class BlockEngineCallback(TrainerCallback):  # type: ignore[misc]
        def __init__(self, opt: Any) -> None:
            self.optimizer = opt

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ARG002
            if drop_input_require_grads_hook(self.optimizer.model):
                logger.info(
                    "block engine: removed the input-requires-grad hook so "
                    "backward stops at the active block."
                )

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ARG002
            drop_input_require_grads_hook(self.optimizer.model)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:  # noqa: ARG002
            self.optimizer.finalize()

    return BlockEngineCallback(optimizer)


# ---------------------------------------------------------------------------
# Fit estimate (a projection until the pod run measures it)
# ---------------------------------------------------------------------------
@dataclass
class BlockEngineFit:
    """Projected peak VRAM for block-coordinate full fine-tuning.

    ``status`` is always a projection label: none of these numbers has been
    measured by this library yet (the RunPod evidence run measures them).
    GB here is 10**9 bytes, to match the paper's ``2M + 16M/D`` formula.
    """

    params_billions: float
    num_blocks: int
    max_block_billions: float
    seq_len: int
    batch_size: int
    paper_formula_gb: float
    weights_gb: float
    active_block_gb: float
    activations_gb: float
    logits_gb: float
    overhead_gb: float
    total_gb: float
    status: str = "projection — not measured"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def estimate_block_engine_vram(
    params_billions: float,
    num_blocks: int,
    seq_len: int,
    batch_size: int,
    *,
    max_block_billions: float | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    vocab_size: int | None = None,
    gradient_checkpointing: bool = True,
    weight_bytes: int = 2,
    overhead_fraction: float = 0.10,
) -> BlockEngineFit:
    """Project peak VRAM for (params, D, seq_len, batch).

    Terms (GB = 1e9 bytes):

    * ``paper_formula_gb`` — BAdam's ``2M + 16M/D`` (Luo et al. 2024), which
      assumes D equal blocks and a separate fp32 master next to the 16-bit
      copy. Reported for reference.
    * ``weights_gb`` — the 16-bit model, ``weight_bytes * M``.
    * ``active_block_gb`` — this implementation holds the active block in fp32
      *instead of* 16-bit (+2 B/param), plus an fp32 gradient (4) and fp32
      AdamW moments (8): **14 B/param of the largest block**. Blocks are not
      equal: with embeddings and an untied head as blocks, the largest block
      is the vocab matrix (e.g. 0.545B of Qwen2.5-7B's 7.6B, vs 0.233B per
      layer), so the peak is set by it, not by M/D.
    * ``activations_gb`` — Korthikanti et al. 2022 (arXiv:2205.05198): about
      ``34·s·b·h`` bytes per layer in 16-bit when attention scores are not
      materialised (SDPA / flash). With gradient checkpointing: the saved
      layer inputs (``2·s·b·h`` per layer) plus one layer's full working set.
    * ``logits_gb`` — ``b·s·V`` × (2 bf16 logits + 4 fp32 upcast in the loss
      + 4 fp32 logit gradient) = 10 bytes.
    * ``overhead_gb`` — ``overhead_fraction`` of the above (allocator
      fragmentation). The CUDA context (~0.5 GB) is outside torch's counters
      and not included.

    When ``hidden_size`` / ``num_layers`` / ``vocab_size`` are unknown the
    activation and logits terms are 0 and a note says so.
    """
    m = float(params_billions)
    d = max(1, int(num_blocks))
    mb = float(max_block_billions) if max_block_billions is not None else m / d
    notes: list[str] = []
    paper = 2.0 * m + 16.0 * m / d
    weights = weight_bytes * m
    active = 14.0 * mb
    acts = 0.0
    if hidden_size and num_layers:
        sbh = float(seq_len) * batch_size * hidden_size
        per_layer_full = 34.0 * sbh / 1e9
        if gradient_checkpointing:
            acts = num_layers * 2.0 * sbh / 1e9 + per_layer_full
        else:
            acts = num_layers * per_layer_full
    else:
        notes.append("hidden_size/num_layers unknown: activations not projected")
    logits = 0.0
    if vocab_size:
        logits = float(batch_size) * seq_len * vocab_size * 10.0 / 1e9
    else:
        notes.append("vocab_size unknown: logits not projected")
    if max_block_billions is None:
        notes.append("max block assumed M/D (equal blocks) — pass max_block_billions")
    sub = weights + active + acts + logits
    over = sub * overhead_fraction
    return BlockEngineFit(
        params_billions=m,
        num_blocks=d,
        max_block_billions=mb,
        seq_len=int(seq_len),
        batch_size=int(batch_size),
        paper_formula_gb=round(paper, 3),
        weights_gb=round(weights, 3),
        active_block_gb=round(active, 3),
        activations_gb=round(acts, 3),
        logits_gb=round(logits, 3),
        overhead_gb=round(over, 3),
        total_gb=round(sub + over, 3),
        notes=notes,
    )


def fit_from_model(
    model: nn.Module, seq_len: int, batch_size: int, *, include_embeddings: bool = True,
    gradient_checkpointing: bool = True,
) -> BlockEngineFit:
    """:func:`estimate_block_engine_vram` with the sizes read off ``model``."""
    part = partition_model(model, include_embeddings=include_embeddings)
    cfg = getattr(model, "config", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    vocab = getattr(cfg, "vocab_size", None)
    return estimate_block_engine_vram(
        part.total_numel / 1e9,
        len(part.blocks),
        seq_len,
        batch_size,
        max_block_billions=part.max_block_numel / 1e9,
        hidden_size=hidden,
        num_layers=layers,
        vocab_size=vocab,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_for_sft_trainer(
    model: nn.Module,
    training_args: Any,
    *,
    switch_block_every: int = DEFAULT_SWITCH_BLOCK_EVERY,
    block_order: str = DEFAULT_BLOCK_ORDER,
    block_writeback: str = DEFAULT_BLOCK_WRITEBACK,
    include_embeddings: bool = True,
) -> tuple[BlockCoordinateOptimizer, Any]:
    """Build ``(optimizer, callback)`` for an HF/TRL trainer from its args.

    AdamW hyperparameters (``learning_rate``, ``adam_beta1/2``,
    ``adam_epsilon``, ``weight_decay``) and the seed come from
    ``training_args``; pass the pair as ``optimizers=(optimizer, None)`` and
    ``callbacks=[callback]``. The trainer then builds its LR scheduler on this
    optimizer, clips the global grad norm over the active block (the only
    parameters with gradients) and counts accumulation as usual.

    A model stored in 16-bit needs the trainer's autocast (``bf16=True`` or
    ``fp16=True``) because the active block is held in fp32; without autocast
    the fp32 weights would meet 16-bit activations in the matmuls.
    """
    stored_low = any(_is_low_precision(p.dtype) for p in model.parameters())
    autocast_on = bool(getattr(training_args, "bf16", False) or getattr(training_args, "fp16", False))
    if stored_low and not autocast_on:
        raise InvalidSettingError(
            setting_name="full_ft_engine",
            value="block",
            expected="bf16=True or fp16=True in the training config for a 16-bit model",
            suggestion=(
                "The block engine holds the active block in fp32 and relies on "
                "the trainer's autocast to run it inside a 16-bit model. Enable "
                "bf16 (Ampere or newer) or fp16 mixed precision."
            ),
        )
    optimizer = BlockCoordinateOptimizer(
        model,
        lr=float(training_args.learning_rate),
        betas=(float(training_args.adam_beta1), float(training_args.adam_beta2)),
        eps=float(training_args.adam_epsilon),
        weight_decay=float(training_args.weight_decay),
        switch_block_every=switch_block_every,
        block_order=block_order,
        block_writeback=block_writeback,
        seed=int(training_args.seed),
        include_embeddings=include_embeddings,
    )
    return optimizer, build_block_engine_callback(optimizer)


def params_by_block(opt: Any) -> Iterable[tuple[str, list[Any]]]:
    """(block name, parameters) pairs — a convenience for tests and receipts."""
    for i, blk in enumerate(opt.partition.blocks):
        yield blk.name, opt.block_params(i)

