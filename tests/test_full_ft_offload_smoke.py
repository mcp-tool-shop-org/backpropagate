"""Real end-to-end FSDP2 CPU-offload full fine-tuning smoke (v1.7 headline).

v1.7's lead claim is single-card 7B-class *full* fine-tuning via
``Trainer(mode="full", full_ft_offload=True)`` / ``--full-ft-offload``: FSDP2
``fully_shard`` + ``CPUOffloadPolicy`` + activation checkpointing + bf16, with a
single-process NCCL group auto-initialized so a bare ``python`` run works (no
``torchrun`` / ``accelerate launch``). Since feat/offload-7b the path runs the
direct-FSDP2 engine (``backpropagate.offload_engine``): bf16 host params, and a
factored Adafactor stepped on the GPU with stochastic rounding.
``tests/test_offload_engine.py`` pins the math on CPU; THIS file is the "the
bytes actually flow through real torch.distributed" proof on a real GPU.

It trains a tiny model (SmolLM2-135M-Instruct) for 2 steps and asserts:

1. The process group was initialized by the trainer itself — NCCL backend,
   world size 1, no ``torchrun`` environment.
2. FSDP2 sharding was actually applied (root + per-block ``FSDPModule``s, params
   are ``DTensor``) and CPU offload actually engaged: the sharded parameters
   live on the CPU, stay bf16 (no fp32 upcast), and the optimizer keeps ~no
   state on the host — not just "no exception was raised".
3. The final loss is finite.
4. ``Trainer.save()`` writes a full-weight checkpoint (not a LoRA adapter) that
   loads back with ``AutoModelForCausalLM.from_pretrained``, differs from the
   base weights (training really updated them), and generates.

It also records peak GPU VRAM (torch allocator) and peak host RSS, printed as a
``OFFLOAD_SMOKE_RECEIPT`` JSON line (run with ``-s`` to see it).

A second test pins the Windows-native fast-fail on the REAL runtime (no mocks):
``full_ft_offload=True`` on a host without NCCL must raise
``DEP_FSDP_UNAVAILABLE`` BEFORE the model is loaded.

Running it (WSL2 / Linux — NCCL is required; Windows-native cannot run it)
--------------------------------------------------------------------------
The venv should live on the Linux filesystem (``/mnt/*`` IO is slow). From a
Windows PowerShell prompt::

    wsl -d Ubuntu -- bash -lc "
      uv venv -p 3.12 ~/bp-offload-smoke &&
      source ~/bp-offload-smoke/bin/activate &&
      uv pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128 &&
      uv pip install -e /mnt/e/AI/backpropagate psutil pytest pytest-timeout"

    wsl -d Ubuntu -- bash -lc "
      source ~/bp-offload-smoke/bin/activate &&
      cd /mnt/e/AI/backpropagate &&
      HF_HUB_CACHE=/mnt/c/Users/mikey/.cache/huggingface/hub \\
      python -m pytest tests/test_full_ft_offload_smoke.py -m 'slow or integration' \\
        -p no:cacheprovider -s -v"

(Point ``HF_HUB_CACHE`` at any cache holding the model, or leave it unset and
let it download ~270 MB.) Blackwell (sm_120, RTX 50xx) needs the cu128 wheels.

Gating
------
* ``@pytest.mark.slow`` AND ``@pytest.mark.integration`` — deselected by the
  fast lane.
* The offload smoke skips (never silently passes) when a training dep is
  missing, CUDA is absent, NCCL is absent (Windows-native — the fast-fail test
  covers that host instead), or the model is unreachable. Every skip names the
  fix.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

# Override to scale the probe up (e.g. HuggingFaceTB/SmolLM2-360M-Instruct) when
# measuring host-RAM growth; the default is the tiny, cache-friendly model. Budget
# host RAM at ~16 bytes/param for the offloaded state (see the receipt).
_SMOKE_MODEL = os.environ.get(
    "BACKPROPAGATE_OFFLOAD_SMOKE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
)


def _model_is_reachable(model_id: str) -> bool:
    """True if ``model_id`` can be loaded — network reachable OR already cached."""
    try:
        # Only the public API: ``try_to_load_from_cache`` returns a str path on a
        # hit (and a non-str sentinel / None otherwise). Do NOT import the
        # private ``_CACHED_NO_EXIST`` from ``huggingface_hub.constants`` — it is
        # not there in huggingface_hub 1.x, and that ImportError silently turned
        # a warm offline cache into "unreachable".
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached, str) and os.path.isfile(cached):
            return True
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _nccl_available() -> bool:
    try:
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_nccl_available())
    except Exception:
        return False


_MISSING_DEPS: list[str] = []
for _dep in ("torch", "trl", "transformers", "accelerate", "datasets"):
    try:
        __import__(_dep)
    except Exception:  # pragma: no cover - environment-dependent
        _MISSING_DEPS.append(_dep)

_SKIP_REASON: str | None = None
if _MISSING_DEPS:
    _SKIP_REASON = (
        f"offload smoke requires {', '.join(_MISSING_DEPS)} (pip install "
        "backpropagate — torch/trl/transformers/accelerate are core deps)"
    )
elif not _cuda_available():
    _SKIP_REASON = (
        "FSDP2 CPU-offload full-FT needs a CUDA GPU. Run this smoke on a CUDA "
        "box under WSL2 / Linux (see the module docstring)."
    )
elif not _nccl_available():
    _SKIP_REASON = (
        "NCCL is unavailable (Windows-native torch). The offload path is Linux / "
        "WSL2 only — run this smoke under WSL2 (see the module docstring). The "
        "Windows-native DEP_FSDP_UNAVAILABLE fast-fail is covered by "
        "test_offload_fast_fails_without_nccl in this file."
    )
elif not _model_is_reachable(_SMOKE_MODEL):
    _SKIP_REASON = (
        f"{_SMOKE_MODEL} is not reachable (no network AND not in the HF cache). "
        f"Run `huggingface-cli download {_SMOKE_MODEL}` or point HF_HUB_CACHE at "
        "a cache that holds it."
    )


# Tiny SFT chat rows (same shape as the FP8 smoke).
_SFT_ROWS: list[dict] = [
    {"messages": [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level, readable programming language."},
    ]},
    {"messages": [
        {"role": "user", "content": "Explain recursion in one sentence."},
        {"role": "assistant", "content": "Recursion is when a function calls itself on a smaller input until a base case."},
    ]},
    {"messages": [
        {"role": "user", "content": "What does HTTP stand for?"},
        {"role": "assistant", "content": "HyperText Transfer Protocol."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]},
    {"messages": [
        {"role": "user", "content": "Name a primary color."},
        {"role": "assistant", "content": "Blue is a primary color."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
    ]},
    {"messages": [
        {"role": "user", "content": "Define an algorithm briefly."},
        {"role": "assistant", "content": "An algorithm is a finite sequence of steps that solves a problem."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is a variable?"},
        {"role": "assistant", "content": "A named container that stores a value in a program."},
    ]},
]


def _write_rows(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in _SFT_ROWS:
            fh.write(json.dumps(row) + "\n")


def _peak_host_rss_gb() -> float | None:
    """Peak resident set size of this process in GiB (Linux ru_maxrss is KiB)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except Exception:
        return None


def _find_fsdp_model(trainer):
    """Return the FSDP2-sharded root module the HF trainer actually trained."""
    from torch.distributed.fsdp import FSDPModule

    candidates = [
        getattr(trainer, "_model", None),
        getattr(getattr(trainer, "_trainer", None), "model", None),
        getattr(getattr(trainer, "_trainer", None), "model_wrapped", None),
    ]
    for cand in candidates:
        if cand is not None and isinstance(cand, FSDPModule):
            return cand
    return candidates[0]


@pytest.fixture
def _clean_process_group():
    """Tear down the process group the trainer lazily creates (process-global)."""
    yield
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(1200)
@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_full_ft_offload_trains_shards_offloads_and_saves(
    tmp_path: Path, _clean_process_group
) -> None:
    """Real FSDP2 CPU-offload full-FT run on a tiny model — see module docstring."""
    import gc

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    if "unsloth" in sys.modules:
        pytest.skip(
            "'unsloth' is already imported in this process and globally patches "
            "the trainer stack; run this smoke in isolation."
        )
    if dist.is_initialized():
        pytest.skip(
            "a torch.distributed process group already exists in this process; "
            "run this smoke in isolation so it proves the trainer's own lazy init."
        )
    # The whole point: no launcher. A torchrun/accelerate-launch env would make
    # the "initialized without torchrun" assertion meaningless.
    assert "TORCHELASTIC_RUN_ID" not in os.environ, (
        "run this smoke with plain `python -m pytest`, not under torchrun."
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from backpropagate.trainer import Trainer, TrainingRun

    data_path = tmp_path / "sft.jsonl"
    _write_rows(data_path)
    output_dir = tmp_path / "offload_output"

    trainer = Trainer(
        model=_SMOKE_MODEL,
        use_unsloth=False,
        mode="full",
        full_ft_offload=True,
        max_seq_length=128,
        batch_size=2,
        gradient_accumulation=1,
        # A larger-than-default LR so 2 steps visibly move bf16 weights (the
        # full-FT 2e-5 default can round away inside bf16 resolution).
        learning_rate=1e-3,
        output_dir=str(output_dir),
        report_to="none",
    )
    assert trainer.mode == "full"
    assert trainer.full_ft_offload is True

    run = trainer.train(str(data_path), steps=2)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    # 1. Process group: created by the trainer, NCCL, single process.
    assert dist.is_initialized(), "trainer did not initialize a process group."
    assert dist.get_backend() == "nccl", f"expected NCCL, got {dist.get_backend()!r}."
    assert dist.get_world_size() == 1

    # 2a. FSDP2 sharding really applied.
    model = _find_fsdp_model(trainer)
    assert isinstance(model, FSDPModule), (
        f"the trained model is not an FSDP2 FSDPModule (got {type(model).__name__}); "
        "full_ft_offload=True did not engage fully_shard."
    )
    fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
    assert len(fsdp_modules) > 1, (
        "only the root is sharded — auto_wrap did not wrap the transformer blocks."
    )
    params = list(model.parameters())
    assert params, "model has no parameters?"
    non_dtensor = [n for n, p in model.named_parameters() if not isinstance(p, DTensor)]
    assert not non_dtensor, f"params not sharded as DTensor: {non_dtensor[:5]}"
    assert all(p.requires_grad for p in params), "full FT must train every parameter."

    # 2b. CPU offload really engaged: the sharded (local) params live on CPU.
    param_devices = {p.to_local().device.type for p in params}
    assert param_devices == {"cpu"}, (
        f"CPUOffloadPolicy did not park the sharded params on CPU: {param_devices}."
    )

    # 2c. Host params stay bf16 (no fp32 upcast), and the optimizer adds ~no
    #     host bytes: the factored Adafactor keeps its row/col state on the GPU.
    #     The v1.7 route stored 16 B/param on the host, the engine ~4.
    assert {p.dtype for p in params} == {torch.bfloat16}, (
        f"offloaded params were upcast: {sorted({str(p.dtype) for p in params})}."
    )
    optimizer = trainer._offload_optimizer
    state_devices: set[str] = set()
    state_dtypes: set[str] = set()
    state_bytes = 0
    host_state_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                local = value.to_local() if isinstance(value, DTensor) else value
                state_devices.add(local.device.type)
                state_dtypes.add(str(local.dtype))
                state_bytes += local.numel() * local.element_size()
                if local.device.type == "cpu":
                    host_state_bytes += local.numel() * local.element_size()
    assert state_devices, "optimizer holds no state after 2 steps — did it step?"

    n_params = sum(p.numel() for p in params)
    param_bytes = sum(p.to_local().numel() * p.to_local().element_size() for p in params)

    assert host_state_bytes / n_params < 0.05, (
        f"optimizer keeps {host_state_bytes / n_params:.2f} B/param on the host."
    )

    # 3. Finite loss.
    assert isinstance(run, TrainingRun)
    assert run.final_loss is not None and math.isfinite(run.final_loss), (
        f"final_loss must be finite; got {run.final_loss!r}."
    )

    # 4. Full-weight artifact saves, loads, changed, generates.
    save_dir = tmp_path / "full_model"
    saved = Path(trainer.save(str(save_dir), run_id=run.run_id))
    assert (saved / "config.json").is_file(), "config.json missing from full-FT save."
    assert not (saved / "adapter_config.json").exists(), (
        "full-FT offload save wrote a LoRA adapter — mode='full' must save full weights."
    )
    weight_files = list(saved.glob("*.safetensors")) + list(saved.glob("*.bin"))
    assert weight_files and all(f.stat().st_size > 0 for f in weight_files)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    reloaded = AutoModelForCausalLM.from_pretrained(str(saved), dtype=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(_SMOKE_MODEL, dtype=torch.float32)
    base_sd = base.state_dict()
    changed = sum(
        1
        for name, tensor in reloaded.state_dict().items()
        if name in base_sd and not torch.equal(tensor, base_sd[name])
    )
    assert changed > 0, "saved weights are identical to the base — training did not update them."

    tok = AutoTokenizer.from_pretrained(str(saved))
    inputs = tok("What is Python?", return_tensors="pt")
    with torch.no_grad():
        out = reloaded.generate(**inputs, max_new_tokens=8, do_sample=False)
    assert out.shape[-1] > inputs["input_ids"].shape[-1], "reloaded model did not generate."

    receipt = {
        "model": _SMOKE_MODEL,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "final_loss": run.final_loss,
        "peak_vram_allocated_gb": round(peak_vram_gb, 3),
        "peak_vram_reserved_gb": round(peak_reserved_gb, 3),
        "peak_host_rss_gb": _peak_host_rss_gb(),
        "fsdp_modules": len(fsdp_modules),
        "param_devices": sorted(param_devices),
        "optimizer": type(optimizer).__name__,
        "optimizer_state_devices": sorted(state_devices),
        # Host-RAM bytes/param of the offloaded training state — the number
        # that decides whether a 7B-class run fits the host (see PR notes).
        "param_dtypes": sorted({str(p.dtype) for p in params}),
        "optimizer_state_dtypes": sorted(state_dtypes),
        "num_params": n_params,
        "host_param_bytes_per_param": round(param_bytes / n_params, 2),
        "host_optim_bytes_per_param": round(state_bytes / n_params, 2),
        "tensors_changed_vs_base": changed,
    }
    print("OFFLOAD_SMOKE_RECEIPT " + json.dumps(receipt))

    # 5. VRAM stays a working set: one unsharded block + activations + the
    #    optimizer's chunked fp32 working set. No phase may hold the model.
    assert peak_vram_gb < 6.0, f"tiny offload run peaked at {peak_vram_gb:.2f} GB."


@pytest.mark.integration
@pytest.mark.skipif(bool(_MISSING_DEPS), reason="needs torch/trl/transformers")
@pytest.mark.skipif(
    _nccl_available(),
    reason="this host HAS NCCL; the no-NCCL fast-fail only fires on Windows-native "
    "(mocked equivalent: test_envelope_v17.py::TestDepFsdp).",
)
def test_offload_fast_fails_without_nccl(tmp_path: Path) -> None:
    """Windows-native (no NCCL): DEP_FSDP_UNAVAILABLE, raised before model load.

    Unmocked — uses the host's real ``torch.distributed``. The mocked twin in
    ``test_envelope_v17.py`` pins the guard function; this pins the wiring:
    ``Trainer.train()`` must call the guard before ``load_model()``, so the
    operator gets the structured error in well under a second instead of a
    multi-GB model load followed by an opaque NCCL failure.
    """
    from backpropagate.exceptions import FsdpUnavailableError
    from backpropagate.trainer import Trainer

    data_path = tmp_path / "sft.jsonl"
    _write_rows(data_path)

    trainer = Trainer(
        model=_SMOKE_MODEL,
        use_unsloth=False,
        mode="full",
        full_ft_offload=True,
        max_seq_length=128,
        output_dir=str(tmp_path / "out"),
        report_to="none",
    )
    with pytest.raises(FsdpUnavailableError) as excinfo:
        trainer.train(str(data_path), steps=2)

    assert excinfo.value.code == "DEP_FSDP_UNAVAILABLE"
    assert trainer._is_loaded is False, "the guard must fire BEFORE the model loads."
    if _cuda_available():
        # CUDA present, NCCL absent: the message must name the way out.
        assert "WSL2" in str(excinfo.value)
