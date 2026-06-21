"""Real end-to-end SimPO training smoke (v1.6 C2 — SimPO Wave 2).

This is the *dogfood proof* for the SimPO feature: a genuine, NON-mocked SimPO
training run that downloads a tiny instruct model, trains ~2 SimPO steps on a
handful of inline preference pairs, and asserts the feature actually works:

* ``Trainer(method="simpo", mode="lora").train(...)`` routes through TRL's
  ``CPOTrainer`` with ``loss_type="simpo"`` + ``cpo_alpha=0.0`` (there is NO
  SimPOTrainer — SimPO is CPOTrainer driven in SimPO mode) and returns a
  :class:`TrainingRun` with a FINITE final loss.
* ``Trainer.save(path)`` writes a real PEFT adapter directory containing
  ``adapter_config.json`` + ``adapter_model.safetensors``, and the adapter is
  loadable via ``PeftModel.from_pretrained`` and mergeable via
  ``merge_and_unload`` (the merge→GGUF→Ollama export path stays intact).
* The run is recorded in ``<output>/run_history.json`` with
  ``hyperparameters.method == "simpo"``.

The trainer-wiring contracts (objective dispatch, the SimPO LR default + clamp,
the simpo+full guard, the paired-dataset path, the OOM-retry rebuild) are
pinned by the trl-MOCKED unit suite in ``tests/test_trainer_simpo_kto.py``.
THIS file is the complementary "the bytes actually flow through real
trl/peft/torch" proof — the only test that spins up a real CPOTrainer in SimPO
mode.

Gating
------
* ``@pytest.mark.slow`` AND ``@pytest.mark.integration`` — the fast CI lane
  (``-m "not gpu and not slow and not integration"``) deselects it.
* ``@pytest.mark.skipif`` — skips cleanly (never a silent pass) when the run
  cannot honestly happen: a required training dep is missing, the tiny model is
  not reachable (no network AND not in the HF cache), or CUDA is unavailable
  (SimPO is a CUDA-rail feature in v1.6). A skip always names the fix.

Proven-good stack
-----------------
Mirrors the ORPO smoke's stack (trl 0.24.0 + transformers 5.x + peft + torch
cu128) on the RTX 5090. The ``warnings_issued`` cross-version shim in
``Trainer._build_trainer`` (provides an inert dict on transformers 5.x) is
applied to the SimPO/CPOTrainer path too, so this smoke RUNS on both
transformers 4.x and 5.x.

Model
-----
``HuggingFaceTB/SmolLM2-135M-Instruct`` — same tiny instruct model the ORPO
smoke uses. A chat template is load-bearing (the prompt/chosen/rejected turns
are rendered through it), so an ``-Instruct`` checkpoint is required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _model_is_reachable(model_id: str) -> bool:
    """True if ``model_id`` can be loaded — network reachable OR already cached."""
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


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_MISSING_DEPS: list[str] = []
for _dep in ("torch", "trl", "transformers", "peft", "datasets"):
    try:
        __import__(_dep)
    except Exception:  # pragma: no cover - environment-dependent
        _MISSING_DEPS.append(_dep)

_SKIP_REASON: str | None = None
if _MISSING_DEPS:
    _SKIP_REASON = (
        f"SimPO smoke requires {', '.join(_MISSING_DEPS)} (install the training "
        "extras: pip install 'backpropagate[unsloth]' or trl+peft+transformers)"
    )
elif not _cuda_available():
    _SKIP_REASON = (
        "SimPO is a CUDA-rail feature in v1.6; this smoke needs a CUDA GPU. "
        "Run it on a CUDA box (e.g. the RTX 5090 rig)."
    )
elif not _model_is_reachable(_SMOKE_MODEL):
    _SKIP_REASON = (
        f"{_SMOKE_MODEL} is not reachable (no network AND not in the HF cache). "
        f"Run `huggingface-cli download {_SMOKE_MODEL}` then re-run "
        "`pytest tests/test_simpo_smoke.py -m 'slow or integration'`."
    )


# Inline preference pairs — SimPO consumes the SAME {prompt, chosen, rejected}
# shape as ORPO (the paired path). Five short rows take ~2 optimizer steps.
_PREFERENCE_ROWS: list[dict[str, str]] = [
    {
        "prompt": "What is Python?",
        "chosen": "Python is a high-level, readable programming language.",
        "rejected": "idk google it",
    },
    {
        "prompt": "Explain recursion in one sentence.",
        "chosen": "Recursion is when a function calls itself on a smaller input until a base case.",
        "rejected": "its when stuff repeats lol",
    },
    {
        "prompt": "What does HTTP stand for?",
        "chosen": "HyperText Transfer Protocol.",
        "rejected": "no idea",
    },
    {
        "prompt": "Give a one-line definition of a variable.",
        "chosen": "A named container that stores a value in a program.",
        "rejected": "a thing",
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "somewhere in europe",
    },
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_simpo_end_to_end_trains_and_saves_adapter(tmp_path: Path) -> None:
    """Real SimPO run: download → train ~2 steps → save adapter → verify.

    NOT mocked — exercises the genuine trl ``CPOTrainer`` in SimPO mode on a
    real (tiny) model. Asserts the load-bearing post-conditions:

    1. ``train()`` returns a ``TrainingRun`` with a FINITE ``final_loss``.
    2. ``save()`` writes a PEFT adapter dir that is LOADABLE
       (``PeftModel.from_pretrained``) and MERGEABLE (``merge_and_unload``).
    3. ``run_history.json`` records the run with ``method == "simpo"``.
    """
    from backpropagate.checkpoints import RunHistoryManager
    from backpropagate.trainer import Trainer, TrainingRun, _SIMPO_DEFAULT_LR

    data_path = tmp_path / "preferences.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in _PREFERENCE_ROWS:
            fh.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "simpo_output"

    trainer = Trainer(
        model=_SMOKE_MODEL,
        use_unsloth=False,
        mode="lora",
        method="simpo",
        lora_r=4,
        max_seq_length=128,
        output_dir=str(output_dir),
        batch_size=1,
        gradient_accumulation=1,
        report_to="none",
    )

    # The SimPO LR default (1e-6) should auto-apply (no explicit LR passed) — a
    # cheap sanity check we're on the SimPO path before spending the download.
    assert trainer.method == "simpo"
    assert trainer.learning_rate == pytest.approx(_SIMPO_DEFAULT_LR)

    run = trainer.train(str(data_path), steps=2)

    # 1. Finite final loss from a real CPOTrainer (SimPO mode).
    assert isinstance(run, TrainingRun), (
        f"train() should return a TrainingRun; got {type(run)!r}."
    )
    assert run.final_loss is not None
    assert math.isfinite(run.final_loss), (
        f"SimPO final_loss must be finite; got {run.final_loss!r}. A non-finite "
        "loss means the CPOTrainer step diverged or never produced a loss."
    )

    # 2. A real, loadable + mergeable PEFT adapter on disk.
    adapter_dir = tmp_path / "adapter"
    saved_path = trainer.save(str(adapter_dir), run_id=run.run_id)
    saved = Path(saved_path)

    assert saved.is_dir(), f"save() should write an adapter directory at {saved}."
    config_file = saved / "adapter_config.json"
    weights_file = saved / "adapter_model.safetensors"
    assert config_file.is_file(), "adapter_config.json missing."
    assert weights_file.is_file(), "adapter_model.safetensors missing."
    assert weights_file.stat().st_size > 0, "adapter_model.safetensors is empty."

    # Load via PeftModel + merge — proves the export path (merge→GGUF→Ollama)
    # is intact for a SimPO adapter.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        _SMOKE_MODEL, torch_dtype=torch.float32
    )
    peft_model = PeftModel.from_pretrained(base, str(saved))
    merged = peft_model.merge_and_unload()
    assert merged is not None, "merge_and_unload returned None for a SimPO adapter."

    # 3. run-history is honestly tagged method='simpo'.
    record = RunHistoryManager(str(output_dir)).get_run(run.run_id)
    assert record is not None, (
        f"run_id {run.run_id} not found in run_history.json under {output_dir}."
    )
    hp = record.get("hyperparameters", {})
    assert hp.get("method") == "simpo", (
        f"run-history must tag method='simpo'; got {hp.get('method')!r}."
    )
