"""Real end-to-end ORPO training smoke (v1.5 T1.2 — ORPO Wave 3).

This is the *dogfood proof* for the ORPO feature: a genuine, NON-mocked ORPO
training run that downloads a tiny instruct model, trains ~2 ORPO steps on a
handful of inline preference pairs, and asserts the feature actually works:

* ``Trainer(method="orpo", mode="lora").train(...)`` routes through TRL's
  ``ORPOTrainer`` and returns a :class:`TrainingRun` with a FINITE final loss.
* ``Trainer.save(path)`` writes a real PEFT adapter directory containing
  ``adapter_config.json`` + ``adapter_model.safetensors``.
* The run is recorded in ``<output>/run_history.json`` with
  ``hyperparameters.method == "orpo"``.

The trainer-wiring contracts (objective dispatch, LR default, the ORPO+full
guard, the dataset preference path, the OOM-retry rebuild) are pinned by the
trl-MOCKED unit suite in ``tests/test_orpo.py``. THIS file is the
complementary "the bytes actually flow through real trl/peft/torch" proof —
deliberately the only test in the suite that spins up an ORPOTrainer for real.

Gating
------
* ``@pytest.mark.slow`` AND ``@pytest.mark.integration`` — so the fast CI lane
  (``-m "not gpu and not slow and not integration"``) deselects it. It runs in
  the nightly / opt-in lane (``-m "slow or integration"``).
* ``@pytest.mark.skipif`` — skips cleanly when the run cannot honestly happen:
  (a) a required training dep is missing, or (b) the tiny model is NOT
  reachable (no network AND not in the local HF cache). A skip here always
  names the fix — it is never a silent pass. (The trl×transformers
  ``warnings_issued`` skew that once forced a skip is now handled inside the
  library — see ``Trainer._build_trainer`` — so this smoke RUNS, and must
  pass, on both transformers 4.x and 5.x.)

Proven-good stack (real runs, RTX 5090, 2026-05-30)
---------------------------------------------------
Trains to a finite loss + writes a real adapter on BOTH the project's target
stack (trl 0.24.0 + transformers 5.5.0 + peft 0.19.1 + torch 2.10.0/cu128) and
the prior stack (trl 0.24.0 + transformers 4.57.x). trl 0.24's ``ORPOTrainer``
writes ``model.warnings_issued`` — an attribute transformers 5.x removed, which
crashed the constructor until v1.5 added a cross-version shim in
``Trainer._build_trainer`` (it provides an inert ``warnings_issued`` dict when
absent). The smoke therefore no longer skips on transformers 5.x; it runs and
must pass.

Model
-----
``HuggingFaceTB/SmolLM2-135M-Instruct`` — a 135M-param instruct model with a
real chat template (ORPO needs an instruct template to render the prompt/
chosen/rejected turns). Small enough to download + train 2 steps in well under
a minute on the RTX 5090; trains on CPU too (the Stage-A
``_detect_optim_for_card`` → ``adamw_torch`` + ``_detect_optimal_dtype`` → fp32
fix makes the ORPO config CPU-constructible), just slower. ``sshleifer/tiny-
gpt2`` was the documented fallback but was NOT needed — SmolLM2-135M-Instruct
downloaded and trained green (see the agent report). It is preferred over
tiny-gpt2 because tiny-gpt2 has no chat template, which ORPO's prompt
formatting depends on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# The tiny instruct model under test. A chat template is load-bearing for ORPO
# (the prompt/chosen/rejected turns are rendered through it), so we use an
# *-Instruct checkpoint rather than a bare base model.
_SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _model_is_reachable(model_id: str) -> bool:
    """True if ``model_id`` can be loaded — network reachable OR already cached.

    The smoke must not fake a pass when the model is unavailable, but it also
    must not require the network when the model is already in the HF cache (the
    common case on a dev box that has run this once). We therefore return True
    if EITHER signal holds:

    * Network: ``HfApi().model_info`` succeeds (the hub is reachable and the
      repo exists / is public).
    * Cache: a core file (``config.json``) is already present in the local HF
      cache via ``try_to_load_from_cache``.

    Any exception on the network probe is swallowed (offline, rate-limited,
    DNS) and we fall back to the cache probe; only if BOTH miss do we report
    unreachable, which drives a clean skip.
    """
    # Cache probe first — cheapest, and the offline-friendly path.
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST

        cached = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached, str) and cached and cached is not _CACHED_NO_EXIST:
            return True
    except Exception:
        pass
    # Network probe.
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return True
    except Exception:
        return False


# NOTE (v1.5): the trl×transformers ``warnings_issued`` skew — trl 0.24's
# ``ORPOTrainer.__init__`` writes ``model.warnings_issued``, an attribute
# transformers 5.x removed, which crashed the constructor before any step — is
# now handled by a cross-version shim in ``Trainer._build_trainer`` (it provides
# an inert dict when the attribute is absent). There is therefore NO static
# stack-incompatibility pre-skip here: the smoke runs on transformers 4.x AND
# 5.x. If a DIFFERENT future trl×transformers skew breaks ORPO, this smoke will
# FAIL (not skip) — the correct, visible signal.

# Resolve the dependency / reachability gates once at import time so the skip
# reason is specific. importorskip can't express "skip if unreachable", so we
# compute a single boolean + reason and feed it to skipif.
_MISSING_DEPS: list[str] = []
for _dep in ("torch", "trl", "transformers", "peft", "datasets"):
    try:
        __import__(_dep)
    except Exception:  # pragma: no cover - environment-dependent
        _MISSING_DEPS.append(_dep)

_SKIP_REASON: str | None = None
if _MISSING_DEPS:
    _SKIP_REASON = (
        f"ORPO smoke requires {', '.join(_MISSING_DEPS)} (install the training "
        "extras: pip install 'backpropagate[unsloth]' or trl+peft+transformers)"
    )
elif not _model_is_reachable(_SMOKE_MODEL):
    _SKIP_REASON = (
        f"{_SMOKE_MODEL} is not reachable (no network AND not in the HF cache). "
        "Run this smoke manually on a box with the model cached: "
        f"`huggingface-cli download {_SMOKE_MODEL}` then re-run "
        "`pytest tests/test_orpo_smoke.py -m 'slow or integration'`."
    )


# Inline preference pairs. ORPO needs {prompt, chosen, rejected} rows: the
# chosen completion is preferred, the rejected one is the dispreferred
# (lower-quality / unhelpful) alternative. Five short rows are enough to take
# ~2 optimizer steps without making the download the bottleneck.
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
def test_orpo_end_to_end_trains_and_saves_adapter(tmp_path: Path) -> None:
    """Real ORPO run: download → train ~2 steps → save adapter → verify.

    This is NOT mocked — it exercises the genuine trl ``ORPOTrainer`` on a
    real (tiny) model. Asserts the three load-bearing post-conditions of the
    ORPO feature:

    1. ``train()`` returns a ``TrainingRun`` with a FINITE ``final_loss``
       (ORPOTrainer actually ran and produced a numeric loss).
    2. ``save()`` writes a PEFT adapter dir with ``adapter_config.json`` +
       ``adapter_model.safetensors`` (a real, loadable LoRA adapter).
    3. ``run_history.json`` records the run with ``method == "orpo"`` (the
       run is honestly tagged as an ORPO run, not SFT).
    """
    from backpropagate.checkpoints import RunHistoryManager
    from backpropagate.trainer import Trainer, TrainingRun

    # --- inputs -----------------------------------------------------------
    data_path = tmp_path / "preferences.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in _PREFERENCE_ROWS:
            fh.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "orpo_output"

    # --- trainer ----------------------------------------------------------
    # use_unsloth=False — Unsloth is not a hard dep and ORPO routes through
    # the transformers/trl backend regardless. mode="lora" is the only ORPO
    # mode in v1.5. lora_r=4 + max_seq_length=128 keep the step tiny.
    trainer = Trainer(
        model=_SMOKE_MODEL,
        use_unsloth=False,
        mode="lora",
        method="orpo",
        lora_r=4,
        max_seq_length=128,
        output_dir=str(output_dir),
        # Keep the loop deterministic + cheap; OOM recovery is irrelevant on a
        # 135M model but harmless if left on.
        batch_size=1,
        gradient_accumulation=1,
        report_to="none",
    )

    # The ORPO LR default (8e-6) should auto-apply since we passed no explicit
    # learning_rate — a cheap sanity check that we're on the ORPO path before
    # we spend the download.
    from backpropagate.trainer import _ORPO_DEFAULT_LR

    assert trainer.method == "orpo"
    assert trainer.learning_rate == pytest.approx(_ORPO_DEFAULT_LR)

    # --- train (REAL) -----------------------------------------------------
    # ~2 optimizer steps. steps= is the hard cap; with 5 rows / batch 1 this
    # is 2 gradient steps, enough to prove the ORPOTrainer loop turns over and
    # emits a finite loss.
    run = trainer.train(str(data_path), steps=2)

    # 1. Finite final loss from a real ORPOTrainer.
    assert isinstance(run, TrainingRun), (
        f"train() should return a TrainingRun; got {type(run)!r}."
    )
    assert run.final_loss is not None
    assert math.isfinite(run.final_loss), (
        f"ORPO final_loss must be finite; got {run.final_loss!r}. A non-finite "
        "loss means the ORPOTrainer step diverged or never produced a loss."
    )

    # --- save (REAL) ------------------------------------------------------
    adapter_dir = tmp_path / "adapter"
    saved_path = trainer.save(str(adapter_dir), run_id=run.run_id)
    saved = Path(saved_path)

    # 2. A real, loadable PEFT adapter on disk.
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
    # The weights file should carry actual tensor bytes, not be a 0-byte stub.
    assert weights_file.stat().st_size > 0, "adapter_model.safetensors is empty."

    # --- run history ------------------------------------------------------
    # 3. The run is recorded and honestly tagged method='orpo'. Run history
    # lives next to the trainer's output_dir (where train() persisted it).
    history = RunHistoryManager(str(output_dir))
    record = history.get_run(run.run_id)
    assert record is not None, (
        f"run_id {run.run_id} not found in run_history.json under {output_dir}."
    )
    hyperparameters = record.get("hyperparameters", {})
    assert hyperparameters.get("method") == "orpo", (
        "run-history must tag the run method='orpo'; got "
        f"{hyperparameters.get('method')!r}."
    )
