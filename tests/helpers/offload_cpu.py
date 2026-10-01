"""CPU-only FSDP2 fixtures for the offload engine tests.

FSDP2 with ``CPUOffloadPolicy`` runs on CPU tensors when the process group is
a single rank, so the engine's real hook wiring (gather, backward, reduce) can
be exercised without a GPU or NCCL. ``init_world`` tries gloo first (a real
one-rank group) and falls back to the ``fake`` backend, then checks that an
FSDP2 forward matches the plain model; a torch whose one-rank gather does not
copy data is skipped rather than trusted.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest
import torch

_STATE: dict[str, Any] = {"initialized": False, "tmp": None}


def _try_init() -> bool:
    import torch.distributed as dist

    if not dist.is_available():
        return False
    tmp = tempfile.mkdtemp(prefix="bp_offload_pg_")
    _STATE["tmp"] = tmp
    path = os.path.join(tmp, "store").replace("\\", "/")
    try:
        dist.init_process_group("gloo", init_method=f"file:///{path}", rank=0, world_size=1)
        return True
    except Exception:  # noqa: BLE001 — gloo cannot resolve a hostname on some hosts
        pass
    try:
        from torch.testing._internal.distributed.fake_pg import FakeStore

        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def cpu_mesh() -> Any:
    """A one-rank CPU device mesh, so FSDP2 never touches a GPU in these tests."""
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


def _fsdp_forward_matches() -> bool:
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    torch.manual_seed(0)
    plain = torch.nn.Linear(8, 8).to(torch.bfloat16)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    expected = plain(x)
    wrapped = torch.nn.Sequential(torch.nn.Linear(8, 8).to(torch.bfloat16))
    wrapped[0].load_state_dict(plain.state_dict())
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    off = CPUOffloadPolicy(pin_memory=False)
    mesh = cpu_mesh()
    fully_shard(wrapped[0], mesh=mesh, mp_policy=mp, offload_policy=off)
    fully_shard(wrapped, mesh=mesh, mp_policy=mp, offload_policy=off)
    return bool(torch.equal(wrapped(x), expected))


def init_world() -> None:
    """Start a one-rank group once per process; skip the calling test if FSDP2 cannot run on CPU."""
    import torch.distributed as dist

    if _STATE["initialized"]:
        return
    try:
        import torch.distributed.fsdp  # noqa: F401
    except ImportError:
        pytest.skip("torch.distributed.fsdp is not available")
    if not (dist.is_initialized() or _try_init()):
        pytest.skip("no one-rank process group available for FSDP2 on CPU")
    _STATE["initialized"] = True
    try:
        ok = _fsdp_forward_matches()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"FSDP2 does not run on CPU here: {exc}")
    if not ok:
        pytest.skip("FSDP2 one-rank gather does not copy data on this torch")


def destroy_world() -> None:
    import torch.distributed as dist

    if _STATE["initialized"] and dist.is_initialized():
        dist.destroy_process_group()
    _STATE["initialized"] = False


def tiny_llama(seed: int = 0, layers: int = 2, dtype: torch.dtype = torch.bfloat16) -> Any:
    """A 2-layer Llama small enough to train a few steps on CPU inside the test timeout."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg).to(dtype)


class ToyTokenizer:
    """Character-level tokenizer over a 62-symbol alphabet; ids 0 (pad) and 1 (eos) are reserved."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, *, truncation: bool = True, max_length: int = 64,
                 add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids = [2 + (ord(c) % 62) for c in text][:max_length]
        return {"input_ids": ids}


class ToyDataset(list):  # type: ignore[type-arg]
    column_names = ["text"]


def toy_dataset(n: int = 8) -> ToyDataset:
    return ToyDataset({"text": f"row {i} says hello world {i * 7}"} for i in range(n))
