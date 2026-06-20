"""Real end-to-end KTO training smoke (v1.6 C2 — KTO Wave 2).

This is the *dogfood proof* for the KTO feature: a genuine, NON-mocked KTO
training run that downloads a tiny instruct model, trains ~2 KTO steps on a
handful of inline UNPAIRED binary-feedback rows, and asserts the feature
actually works:

* ``Trainer(method="kto", mode="lora").train(...)`` routes through TRL's
  ``KTOTrainer`` (with NO explicit ``ref_model`` — the frozen LoRA base IS the
  reference, so the 16GB envelope is preserved) and returns a
  :class:`TrainingRun` with a FINITE final loss.
* ``Trainer.save(path)`` writes a real PEFT adapter directory that is loadable
  via ``PeftModel.from_pretrained`` and mergeable via ``merge_and_unload``.
* The run is recorded in ``<output>/run_history.json`` with
  ``hyperparameters.method == "kto"``.

The trainer-wiring contracts (objective dispatch, the KTO LR default + clamp,
the kto+full guard, the unpaired-dataset path, the no-ref_model contract, the
auto-weighting band, the OOM-retry rebuild) are pinned by the trl-MOCKED unit
suite in ``tests/test_trainer_simpo_kto.py``. THIS file is the complementary
"the bytes actually flow through real trl/peft/torch" proof.

Gating
------
* ``@pytest.mark.slow`` AND ``@pytest.mark.integration`` — deselected by the
  fast lane.
* ``@pytest.mark.skipif`` — skips cleanly (never a silent pass) on a missing
  training dep, an unreachable model, or no CUDA (KTO is a CUDA-rail feature in
  v1.6). A skip always names the fix.

Data
----
KTO needs UNPAIRED ``{prompt, completion, label:bool}`` rows. We include BOTH
desirable (label=True) and undesirable (label=False) examples — KTO's KL
estimate needs both polarities, and the run would warn/fail on one-sided data.
The set is deliberately label-imbalanced (4 desirable, 2 undesirable) so the
trainer's auto-weighting visibly rebalances the polarity weights into the
[1:1, 4:3] band; the test asserts the rebalance fired.

Model
-----
``HuggingFaceTB/SmolLM2-135M-Instruct`` — same tiny instruct model the ORPO /
SimPO smokes use (a chat template is load-bearing for rendering the completion).
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
        f"KTO smoke requires {', '.join(_MISSING_DEPS)} (install the training "
        "extras: pip install 'backpropagate[unsloth]' or trl+peft+transformers)"
    )
elif not _cuda_available():
    _SKIP_REASON = (
        "KTO is a CUDA-rail feature in v1.6; this smoke needs a CUDA GPU. Run "
        "it on a CUDA box (e.g. the RTX 5090 rig)."
    )
elif not _model_is_reachable(_SMOKE_MODEL):
    _SKIP_REASON = (
        f"{_SMOKE_MODEL} is not reachable (no network AND not in the HF cache). "
        f"Run `huggingface-cli download {_SMOKE_MODEL}` then re-run "
        "`pytest tests/test_kto_smoke.py -m 'slow or integration'`."
    )


# Inline KTO rows: {prompt, completion, label}. 4 desirable + 2 undesirable —
# imbalanced on purpose so auto-weighting visibly rebalances toward [1:1, 4:3].
_KTO_ROWS: list[dict[str, object]] = [
    {"prompt": "What is Python?", "completion": "A readable programming language.", "label": True},
    {"prompt": "What does HTTP stand for?", "completion": "HyperText Transfer Protocol.", "label": True},
    {"prompt": "Define a variable.", "completion": "A named container for a value.", "label": True},
    {"prompt": "Capital of France?", "completion": "Paris.", "label": True},
    {"prompt": "What is Python?", "completion": "idk google it", "label": False},
    {"prompt": "Capital of France?", "completion": "somewhere in europe", "label": False},
]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_kto_end_to_end_trains_and_saves_adapter(tmp_path: Path) -> None:
    """Real KTO run: download → train ~2 steps → save adapter → verify.

    NOT mocked — exercises the genuine trl ``KTOTrainer`` on a real (tiny)
    model. Asserts the load-bearing post-conditions:

    1. ``train()`` returns a ``TrainingRun`` with a FINITE ``final_loss``.
    2. Auto-weighting fired — the resolved KTO weights are populated (and the
       imbalanced data nudged at least one weight off its 1.0 seed).
    3. ``save()`` writes a PEFT adapter dir that is LOADABLE + MERGEABLE.
    4. ``run_history.json`` records the run with ``method == "kto"``.
    """
    from backpropagate.checkpoints import RunHistoryManager
    from backpropagate.trainer import Trainer, TrainingRun, _KTO_DEFAULT_LR

    data_path = tmp_path / "kto.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in _KTO_ROWS:
            fh.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "kto_output"

    trainer = Trainer(
        model=_SMOKE_MODEL,
        use_unsloth=False,
        mode="lora",
        method="kto",
        lora_r=4,
        max_seq_length=128,
        output_dir=str(output_dir),
        # KTO's in-batch KL estimate requires an ACTUAL batch size > 1 (TRL
        # rejects batch 1). The trainer auto-bumps a batch_size=1 KTO run to 2;
        # we set it explicitly here to document the requirement.
        batch_size=2,
        gradient_accumulation=1,
        report_to="none",
    )

    # The KTO LR default (1e-6) should auto-apply (no explicit LR) — sanity
    # check we're on the KTO path before spending the download.
    assert trainer.method == "kto"
    assert trainer.learning_rate == pytest.approx(_KTO_DEFAULT_LR)

    run = trainer.train(str(data_path), steps=2)

    # 1. Finite final loss from a real KTOTrainer.
    assert isinstance(run, TrainingRun), (
        f"train() should return a TrainingRun; got {type(run)!r}."
    )
    assert run.final_loss is not None
    assert math.isfinite(run.final_loss), (
        f"KTO final_loss must be finite; got {run.final_loss!r}. A non-finite "
        "loss means the KTOTrainer step diverged or never produced a loss."
    )

    # 2. Auto-weighting fired: resolved weights populated, and the 4:2 imbalance
    # (seed ratio 2.0 > 4/3) scaled the undesirable weight up off its 1.0 seed.
    assert hasattr(trainer, "_kto_resolved_desirable_weight")
    assert hasattr(trainer, "_kto_resolved_undesirable_weight")
    d = trainer._kto_resolved_desirable_weight
    u = trainer._kto_resolved_undesirable_weight
    assert d > 0 and u > 0
    assert u > 1.0, (
        "auto-weighting should have scaled the undesirable weight above its "
        f"1.0 seed for the 4-desirable:2-undesirable split; got u={u!r}."
    )

    # 3. A real, loadable + mergeable PEFT adapter on disk.
    adapter_dir = tmp_path / "adapter"
    saved_path = trainer.save(str(adapter_dir), run_id=run.run_id)
    saved = Path(saved_path)

    assert saved.is_dir(), f"save() should write an adapter directory at {saved}."
    assert (saved / "adapter_config.json").is_file(), "adapter_config.json missing."
    weights_file = saved / "adapter_model.safetensors"
    assert weights_file.is_file(), "adapter_model.safetensors missing."
    assert weights_file.stat().st_size > 0, "adapter_model.safetensors is empty."

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        _SMOKE_MODEL, torch_dtype=torch.float32
    )
    peft_model = PeftModel.from_pretrained(base, str(saved))
    merged = peft_model.merge_and_unload()
    assert merged is not None, "merge_and_unload returned None for a KTO adapter."

    # 4. run-history is honestly tagged method='kto'.
    record = RunHistoryManager(str(output_dir)).get_run(run.run_id)
    assert record is not None, (
        f"run_id {run.run_id} not found in run_history.json under {output_dir}."
    )
    hp = record.get("hyperparameters", {})
    assert hp.get("method") == "kto", (
        f"run-history must tag method='kto'; got {hp.get('method')!r}."
    )
