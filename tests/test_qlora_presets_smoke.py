"""Real-GPU QLoRA smoke for the four v1.7 "32 GB envelope" presets.

v1.7 ships four QLoRA presets whose descriptions carry measured-sounding VRAM
numbers — ``llama-3.1-8b`` (~7-8 GB), ``qwen2.5-14b`` (~8.5 GB),
``mistral-small-24b`` (~18 GB at max_seq 4096) and ``qwen2.5-32b`` (~26 GB, and
"JUST fits" only at max_seq 2048). ``tests/test_envelope_v17.py`` only checks
the presets exist. THIS file executes them on the card:

* ``llama-3.1-8b`` / ``qwen2.5-14b`` — train 2 real QLoRA steps end-to-end and
  save a PEFT adapter whose rank matches the preset.
* ``mistral-small-24b`` / ``qwen2.5-32b`` — load in 4-bit and run a 1-step
  no-OOM probe at the preset's SHIPPED ``recommended_max_seq_length``. The step
  input is proven (by a ``training_step`` spy) to be exactly that many tokens,
  and ``oom_recovery=False`` makes an OOM fail loudly instead of being silently
  retried at a smaller shape.

Each run prints a ``PRESET_SMOKE_RECEIPT`` JSON line (peak VRAM, loss, step
input lengths) — run with ``-s`` and paste the receipts into the release evidence.

Presets are advisory: ``Trainer`` does NOT apply ``recommended_lora_r`` /
``recommended_max_seq_length`` / ``recommended_packing`` on its own (its own
defaults are the LoRA "quality" overlay). This smoke passes them explicitly —
exactly what an operator following the preset must do — so it measures the
configuration the preset documents.

Opt-in only
-----------
This needs the WHOLE 32 GB card and ~160 GB of downloads (8B ~16 GB, 14B
~30 GB, 24B ~48 GB, 32B ~66 GB of bf16 safetensors; quantized to 4-bit at
load). It never runs unless explicitly requested::

    BACKPROPAGATE_RUN_PRESET_SMOKE=1

Without that variable every case skips with the instruction. Run the presets
serially, on an otherwise idle GPU (the smoke also skips a case whose free VRAM
is below the card-size floor). ``meta-llama/Llama-3.1-8B-Instruct`` is gated:
accept the licence on the Hub and ``huggingface-cli login`` first, or that case
skips.

Invocation (Windows-native works — QLoRA needs no NCCL; WSL2 works too)::

    huggingface-cli download Qwen/Qwen2.5-14B-Instruct
    huggingface-cli download mistralai/Mistral-Small-24B-Instruct-2501
    huggingface-cli download Qwen/Qwen2.5-32B-Instruct
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

    # PowerShell
    $env:BACKPROPAGATE_RUN_PRESET_SMOKE = "1"
    python -m pytest tests/test_qlora_presets_smoke.py -m "slow or integration" -s -v -p no:randomly

    # bash / WSL2
    BACKPROPAGATE_RUN_PRESET_SMOKE=1 python -m pytest tests/test_qlora_presets_smoke.py \\
        -m 'slow or integration' -s -v -p no:randomly

Select one preset with ``-k qwen2.5-32b``. Under WSL2 mind the VM memory cap
(``.wslconfig``): 4-bit loading streams one bf16 shard at a time, so ~8 GB of
host RAM headroom is enough.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

_OPT_IN_ENV = "BACKPROPAGATE_RUN_PRESET_SMOKE"

# (preset name, mode, steps, minimum card VRAM in GiB to attempt it).
# "train" = 2 steps + adapter save; "probe" = 1 step at the shipped max_seq.
_CASES: list[tuple[str, str, int, float]] = [
    ("llama-3.1-8b", "train", 2, 15.0),
    ("qwen2.5-14b", "train", 2, 15.0),
    ("mistral-small-24b", "probe", 1, 30.0),
    ("qwen2.5-32b", "probe", 1, 30.0),
]

_GIB = 1024 ** 3


def _opted_in() -> bool:
    return os.environ.get(_OPT_IN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _missing_deps() -> list[str]:
    missing = []
    for dep in ("torch", "trl", "transformers", "peft", "datasets", "bitsandbytes"):
        try:
            __import__(dep)
        except Exception:  # pragma: no cover - environment-dependent
            missing.append(dep)
    return missing


def _model_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(model_id, "config.json")
        return isinstance(hit, str) and os.path.isfile(hit)
    except Exception:
        return False


def _model_access_problem(model_id: str) -> str | None:
    """None if the weights are cached or downloadable; else why not."""
    if _model_cached(model_id):
        return None
    try:
        from huggingface_hub import auth_check

        auth_check(model_id)
        return None
    except Exception as exc:  # GatedRepoError, RepositoryNotFoundError, offline
        return f"{type(exc).__name__}: {exc}".splitlines()[0]


def _rows(n_rows: int, max_seq: int) -> list[dict]:
    """Deterministic chat rows each LONGER than ``max_seq`` tokens.

    Without packing, an over-length row is truncated to exactly ``max_seq``, so
    every training example — whichever one the sampler picks for the single
    probe step — has the shipped shape. (With packing, TRL's BFD packer splits
    over-length rows and leaves short tail bins — measured on SmolLM2 at 2048:
    9 bins of 2048 plus tails of 1737-1754 — so a 1-step probe would hit a
    short bin ~1 time in 4. Short rows are worse: every bin lands at 1.7-1.8K.)
    """
    topics = [
        "binary search", "hash tables", "TCP handshakes", "garbage collection",
        "B-trees", "unicode normalization", "rate limiting", "consistent hashing",
    ]
    # ~25-30 tokens per sentence on the preset tokenizers; 2x max_seq of text.
    n_sentences = (2 * max_seq) // 25
    rows = []
    for i in range(n_rows):
        topic = topics[i % len(topics)]
        answer = " ".join(
            f"Point {j + 1} about {topic}: it trades memory for time in case {i}-{j}, "
            f"and the invariant that makes it correct must hold after every update."
            for j in range(n_sentences)
        )
        rows.append({"messages": [
            {"role": "user", "content": f"Explain {topic} in depth (variant {i})."},
            {"role": "assistant", "content": answer},
        ]})
    return rows


def _case_params():
    return [pytest.param(name, mode, steps, floor, id=name) for name, mode, steps, floor in _CASES]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("preset_name,kind,steps,vram_floor_gb", _case_params())
def test_qlora_preset_fits_the_envelope(
    preset_name: str,
    kind: str,
    steps: int,
    vram_floor_gb: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _opted_in():
        pytest.skip(
            f"opt-in only: set {_OPT_IN_ENV}=1 to run the 8B-32B preset smoke "
            "(needs the whole 32 GB card and ~160 GB of downloads)."
        )
    missing = _missing_deps()
    if missing:
        pytest.skip(f"preset smoke requires {', '.join(missing)}.")
    if "unsloth" in sys.modules:
        pytest.skip("'unsloth' is imported in this process; run the smoke in isolation.")

    import gc

    import torch

    if not torch.cuda.is_available():
        pytest.skip("QLoRA presets need a CUDA GPU.")
    free_b, total_b = torch.cuda.mem_get_info()
    total_gb, free_gb = total_b / _GIB, free_b / _GIB
    if total_gb < vram_floor_gb:
        pytest.skip(f"{preset_name} targets a >= {vram_floor_gb:.0f} GB card; this one has {total_gb:.1f} GB.")
    if free_gb < vram_floor_gb:
        pytest.skip(
            f"only {free_gb:.1f} GB of {total_gb:.1f} GB VRAM is free — another process "
            f"is using the GPU. {preset_name} needs >= {vram_floor_gb:.0f} GB free; "
            "rerun on an idle card."
        )

    from backpropagate.config import MODEL_PRESETS
    from backpropagate.trainer import Trainer, TrainingRun

    preset = MODEL_PRESETS[preset_name]
    problem = _model_access_problem(preset.model_id)
    if problem is not None:
        pytest.skip(
            f"{preset.model_id} is not cached and not downloadable ({problem}). "
            f"Run `huggingface-cli download {preset.model_id}` (gated repos: accept "
            "the licence and `huggingface-cli login` first)."
        )

    max_seq = preset.recommended_max_seq_length
    # Probes: packing OFF + over-length rows => every step input is EXACTLY
    # max_seq tokens (see _rows). At batch 1 that is the same tensor shape a
    # full packed bin produces, so the no-OOM result covers the shipped shape
    # deterministically. Train cases keep the preset's packing setting.
    packing = False if kind == "probe" else preset.recommended_packing
    n_rows = 4 * steps
    data_path = tmp_path / "sft.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in _rows(n_rows, max_seq):
            fh.write(json.dumps(row) + "\n")

    # Spy on the real step inputs: the sequence length the GPU actually saw.
    import trl

    step_lengths: list[int] = []
    _orig_step = trl.SFTTrainer.training_step

    def _spy_step(self, model, inputs, *args, **kwargs):
        ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if ids is not None:
            # padding-free packing flattens to (1, total); unpacked is (B, L).
            step_lengths.append(int(ids.shape[-1]))
        return _orig_step(self, model, inputs, *args, **kwargs)

    monkeypatch.setattr(trl.SFTTrainer, "training_step", _spy_step)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(
        model=preset.model_id,
        use_unsloth=False,
        mode="lora",
        lora_r=preset.recommended_lora_r,
        lora_alpha=preset.recommended_lora_r,  # presets document alpha == rank
        max_seq_length=max_seq,
        packing=packing,
        batch_size=1,
        gradient_accumulation=1,
        oom_recovery=False,  # an OOM must FAIL, not retry at a smaller shape
        output_dir=str(tmp_path / "out"),
        report_to="none",
    )
    run = trainer.train(str(data_path), steps=steps)

    peak_alloc_gb = torch.cuda.max_memory_allocated() / _GIB
    peak_reserved_gb = torch.cuda.max_memory_reserved() / _GIB

    assert isinstance(run, TrainingRun)
    assert run.final_loss is not None and math.isfinite(run.final_loss), (
        f"{preset_name}: final_loss must be finite; got {run.final_loss!r}."
    )

    # Did the step actually run at the shipped max_seq? Otherwise "no OOM" would
    # have been measured at a shorter, cheaper shape.
    assert step_lengths, "the training_step spy saw no inputs — did any step run?"
    max_step_len = max(step_lengths)
    if kind == "probe":
        assert max_step_len == max_seq, (
            f"{preset_name}: the probe step ran at {max_step_len} tokens, not the "
            f"shipped max_seq_length={max_seq} — the no-OOM result would not cover "
            "the documented shape."
        )

    adapter_rank = None
    if kind == "train":
        saved = Path(trainer.save(str(tmp_path / "adapter"), run_id=run.run_id))
        cfg_path = saved / "adapter_config.json"
        assert cfg_path.is_file(), f"{preset_name}: adapter_config.json missing."
        weights = saved / "adapter_model.safetensors"
        assert weights.is_file() and weights.stat().st_size > 0
        adapter_rank = json.loads(cfg_path.read_text(encoding="utf-8")).get("r")
        assert adapter_rank == preset.recommended_lora_r, (
            f"{preset_name}: adapter rank {adapter_rank} != preset rank "
            f"{preset.recommended_lora_r}."
        )

    receipt = {
        "preset": preset_name,
        "model": preset.model_id,
        "kind": kind,
        "steps": steps,
        "gpu": torch.cuda.get_device_name(0),
        "card_total_gb": round(total_gb, 2),
        "max_seq_length": max_seq,
        "packing": packing,
        "step_input_lengths": step_lengths,
        "lora_r": preset.recommended_lora_r,
        "adapter_rank": adapter_rank,
        "final_loss": run.final_loss,
        "peak_vram_allocated_gb": round(peak_alloc_gb, 2),
        "peak_vram_reserved_gb": round(peak_reserved_gb, 2),
        "preset_description": preset.description,
    }
    print("PRESET_SMOKE_RECEIPT " + json.dumps(receipt))

    # The envelope claim: the preset fits the card it targets.
    assert peak_reserved_gb < total_gb, (
        f"{preset_name}: reserved {peak_reserved_gb:.1f} GB >= card {total_gb:.1f} GB."
    )

    # Free the card for the next (serial) preset.
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
