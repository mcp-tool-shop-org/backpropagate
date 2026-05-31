"""Real end-to-end FP8 training smoke (v1.5 T2.1 — FP8 compute path).

This is the *dogfood proof* for the FP8 feature: a genuine, NON-mocked LoRA
training run with ``fp8=True`` that downloads a tiny instruct model, converts the
BASE projection linears to torchao ``Float8Linear`` (the LoRA adapter, lm_head,
and embeddings excluded), trains ~2 SFT steps, and asserts the feature actually
works AND the load-bearing exclusion holds:

* ``Trainer(fp8=True, mode='lora', method='sft').train(...)`` returns a
  :class:`TrainingRun` with a FINITE final loss (FP8 backward did not diverge or
  crash — the rank-r adapter exclusion is what makes backward survive).
* ≥1 BASE module is a ``Float8Linear`` AND ZERO ``lora_*`` modules are — the
  exact contract the ``_fp8_module_filter`` predicate enforces, proven on a real
  PEFT-wrapped model rather than a mock.
* ``save()`` writes a real PEFT adapter dir (``adapter_config.json`` +
  non-empty ``adapter_model.safetensors``).
* The adapter MERGES (``merge_and_unload`` / ``save(save_merged=True)``) — the
  mergeability proof (FP8 keeps the adapter exportable to GGUF→Ollama, unlike
  the killed AQLM-2bit path).

The trainer-wiring contracts (the gate ladder, the filter predicate, graceful
fallback, ``_fp8_supported`` truth table) are pinned by the torchao-MOCKED unit
suite in ``tests/test_fp8.py``. THIS file is the complementary "the bytes
actually flow through real torchao float8 + trl + peft + torch" proof.

Gating (mirrors tests/test_orpo_smoke.py)
-----------------------------------------
* ``@pytest.mark.slow`` AND ``@pytest.mark.integration`` — the fast CI lane
  (``-m "not gpu and not slow and not integration"``) deselects it; it runs in
  the nightly / opt-in lane (``-m "slow or integration"``).
* ``@pytest.mark.skipif`` — skips cleanly, NEVER a silent pass, when the run
  cannot honestly happen. The skip ladder names the exact fix for each axis:
  - a required training dep is missing (torch/trl/transformers/peft/datasets/
    torchao) → ``pip install 'backpropagate[fp8]'``;
  - no CUDA (FP8 has no CPU kernel);
  - compute capability < 9 (FP8 needs Hopper sm_90+ / Blackwell sm_120);
  - the tiny model is not reachable (no network AND not in the HF cache).

Model
-----
``HuggingFaceTB/SmolLM2-135M-Instruct`` — same tiny instruct checkpoint the ORPO
smoke uses (real chat template, downloads + trains 2 steps in well under a minute
on the RTX 5090 / sm_120, the FP8-verified card). FP8 layers on top of a bf16
unquantized base; ``load_in_4bit`` is auto-flipped off by the gate ladder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# The tiny instruct model under test (shared with the ORPO smoke).
_SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _model_is_reachable(model_id: str) -> bool:
    """True if ``model_id`` can be loaded — network reachable OR already cached.

    Copied verbatim from tests/test_orpo_smoke.py: cache probe first (offline-
    friendly), then a network probe; only if BOTH miss is the model reported
    unreachable, which drives a clean skip rather than a fake pass.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST

        cached = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached, str) and cached and cached is not _CACHED_NO_EXIST:
            return True
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return True
    except Exception:
        return False


def _fp8_capable() -> tuple[bool, str | None]:
    """Resolve the CUDA + compute-capability axes once for the skip ladder.

    torchao presence is handled by the dep loop below (it is an importable
    package); this covers the two HARDWARE axes the dep loop can't express:
    CUDA must be available, and the GPU must be Hopper (sm_90) or newer. Returns
    ``(True, None)`` when both hold, else ``(False, reason)``.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - the dep loop already flags this
        return False, "torch is not importable"
    if not torch.cuda.is_available():
        return False, (
            "no CUDA device — FP8 tensor-core ops have no CPU kernel. Run this "
            "smoke on a Hopper/Blackwell GPU."
        )
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"could not query CUDA compute capability ({exc!r})"
    if major < 9:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover
            name = "this GPU"
        return False, (
            f"GPU compute capability sm_{major}x ({name}) < sm_90; FP8 needs "
            "Hopper (sm_90) or Blackwell (sm_120). Run on an H100 / RTX 50-series."
        )
    return True, None


# Resolve the dependency / hardware / reachability gates once at import time so
# the skip reason is specific (importorskip can't express the hardware checks).
_MISSING_DEPS: list[str] = []
for _dep in ("torch", "trl", "transformers", "peft", "datasets", "torchao"):
    try:
        __import__(_dep)
    except Exception:  # pragma: no cover - environment-dependent
        _MISSING_DEPS.append(_dep)

_SKIP_REASON: str | None = None
if _MISSING_DEPS:
    _SKIP_REASON = (
        f"FP8 smoke requires {', '.join(_MISSING_DEPS)} — install the FP8 "
        "extras: pip install 'backpropagate[fp8]'."
    )
else:
    _capable, _cap_reason = _fp8_capable()
    if not _capable:
        _SKIP_REASON = f"FP8 smoke: {_cap_reason}"
    elif not _model_is_reachable(_SMOKE_MODEL):
        _SKIP_REASON = (
            f"{_SMOKE_MODEL} is not reachable (no network AND not in the HF "
            "cache). Cache it first: "
            f"`huggingface-cli download {_SMOKE_MODEL}` then re-run "
            "`pytest tests/test_fp8_smoke.py -m 'slow or integration'`."
        )


# A handful of tiny SFT rows (ShareGPT-ish chat). FP8 SFT trains on a single
# ``text`` column rendered through the chat template — no preference pairs.
_SFT_ROWS: list[dict] = [
    {
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {
                "role": "assistant",
                "content": "Python is a high-level, readable programming language.",
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain recursion in one sentence."},
            {
                "role": "assistant",
                "content": (
                    "Recursion is when a function calls itself on a smaller "
                    "input until a base case."
                ),
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What does HTTP stand for?"},
            {"role": "assistant", "content": "HyperText Transfer Protocol."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Give a one-line definition of a variable."},
            {
                "role": "assistant",
                "content": "A named container that stores a value in a program.",
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Name a primary color."},
            {"role": "assistant", "content": "Blue is a primary color."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Define an algorithm briefly."},
            {
                "role": "assistant",
                "content": "An algorithm is a finite sequence of steps that solves a problem.",
            },
        ]
    },
]


def _count_float8_modules(model) -> tuple[int, int]:
    """Return ``(base_float8, lora_float8)`` counts over ``model.named_modules``.

    ``base_float8`` = Float8Linear modules whose FQN is NOT a ``lora_`` adapter.
    ``lora_float8`` = Float8Linear modules whose FQN IS a ``lora_`` adapter (this
    MUST be 0 — converting the rank-r adapter linears is the backward-crash bug
    the module filter prevents).
    """
    from torchao.float8.float8_linear import Float8Linear

    base = 0
    lora = 0
    for name, module in model.named_modules():
        if isinstance(module, Float8Linear):
            if "lora_" in name.lower():
                lora += 1
            else:
                base += 1
    return base, lora


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_fp8_end_to_end_trains_converts_and_merges(tmp_path: Path) -> None:
    """Real FP8 LoRA run: download → convert base to Float8Linear → train →
    save → merge. The four load-bearing post-conditions of the FP8 feature.

    1. ``train()`` returns a ``TrainingRun`` with a FINITE ``final_loss`` (FP8
       backward survived — the rank-r adapter exclusion held).
    2. ``self._fp8_effective is True`` AND ≥1 base module is ``Float8Linear``
       AND ZERO ``lora_*`` modules are (the filter contract, on a real model).
    3. ``save()`` writes a loadable PEFT adapter (config + non-empty weights).
    4. The adapter MERGES into the base (mergeability proof — FP8 keeps the
       export-to-GGUF path open).
    """
    from backpropagate.trainer import Trainer, TrainingRun

    # --- inputs -----------------------------------------------------------
    data_path = tmp_path / "sft.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in _SFT_ROWS:
            fh.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "fp8_output"

    # --- trainer ----------------------------------------------------------
    # use_unsloth=False — FP8 forces the transformers backend anyway (the loader
    # forces it with an INFO log if Unsloth is on). mode='lora' + method='sft'
    # are the ONLY FP8-supported combination in v1.5. lora_r=16 + max_seq=128
    # keep the step tiny. batch_size=8 matches the kickoff spec.
    trainer = Trainer(
        model=_SMOKE_MODEL,
        use_unsloth=False,
        mode="lora",
        method="sft",
        fp8=True,
        lora_r=16,
        max_seq_length=128,
        output_dir=str(output_dir),
        batch_size=8,
        gradient_accumulation=1,
        report_to="none",
    )

    # The gate ladder should have resolved FP8 as effective on this card
    # (CUDA + torchao + sm>=9 all confirmed by the skip ladder above).
    assert trainer.fp8 is True
    assert trainer._fp8_effective is True, (
        "FP8 was requested on an FP8-capable host (the skip ladder confirmed "
        "CUDA + torchao + sm>=9) but _fp8_effective is False — the gate ladder "
        "degraded when it should not have."
    )
    # FP8 supersedes the default 4-bit quantization.
    assert trainer._load_in_4bit is False

    # --- load model (REAL: download + LoRA attach + Float8Linear convert) --
    trainer.load_model()

    # 2. The conversion contract on a REAL PEFT-wrapped model.
    base_f8, lora_f8 = _count_float8_modules(trainer._model)
    assert base_f8 >= 1, (
        f"expected >=1 base Float8Linear after conversion; got {base_f8}. The "
        "FP8 conversion did not bite (filter matched no base projection linear)."
    )
    assert lora_f8 == 0, (
        f"expected ZERO lora_* Float8Linear modules; got {lora_f8}. Converting "
        "the rank-r adapter linears is the backward-crash bug the module filter "
        "exists to prevent — this is the load-bearing exclusion."
    )
    # _fp8_effective must still be True after a successful conversion.
    assert trainer._fp8_effective is True

    # --- train (REAL) -----------------------------------------------------
    run = trainer.train(str(data_path), steps=2)

    # 1. Finite final loss from a real FP8 SFT step.
    assert isinstance(run, TrainingRun), (
        f"train() should return a TrainingRun; got {type(run)!r}."
    )
    assert run.final_loss is not None
    assert math.isfinite(run.final_loss), (
        f"FP8 final_loss must be finite; got {run.final_loss!r}. A non-finite "
        "loss means the FP8 step diverged or the scaled_mm backward failed."
    )

    # --- save (REAL) ------------------------------------------------------
    adapter_dir = tmp_path / "adapter"
    saved_path = trainer.save(str(adapter_dir), run_id=run.run_id)
    saved = Path(saved_path)

    # 3. A real, loadable PEFT adapter on disk.
    assert saved.is_dir(), f"save() should write an adapter directory at {saved}."
    config_file = saved / "adapter_config.json"
    weights_file = saved / "adapter_model.safetensors"
    assert config_file.is_file(), (
        f"adapter_config.json missing from {saved} — PEFT did not write the "
        "LoRA adapter config."
    )
    assert weights_file.is_file(), (
        f"adapter_model.safetensors missing from {saved} — PEFT did not write "
        "the LoRA adapter weights."
    )
    assert weights_file.stat().st_size > 0, "adapter_model.safetensors is empty."

    # 4. Mergeability proof: the adapter merges back into the base. This is the
    # FP8-keeps-export-open contract (the killed AQLM-2bit path could NOT merge).
    # merge_and_unload is the canonical PEFT merge; it must not raise.
    merged = trainer._model.merge_and_unload()
    assert merged is not None, (
        "merge_and_unload() returned None — the FP8 LoRA adapter did not merge, "
        "which would break the GGUF→Ollama export path FP8 is supposed to keep."
    )
