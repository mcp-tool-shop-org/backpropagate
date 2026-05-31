"""Real end-to-end MLX (Apple-Silicon) training smoke (v1.5 T3.1 — MLX rail).

This is the *dogfood proof* for the MLX backend: a genuine, NON-mocked LoRA SFT
run on Apple's ``mlx_lm.lora`` toolchain that trains ~2 steps on a handful of
inline chat rows and asserts the feature actually works:

* ``Trainer(backend="mlx", mode="lora", method="sft").train(...)`` routes
  through :meth:`backpropagate.trainer.Trainer._train_with_mlx` (which drives
  ``mlx_lm.lora`` via the subprocess seam) and returns a :class:`TrainingRun`
  with a FINITE ``final_loss``.
* The run writes a real adapter — at least one ``*.safetensors`` file under the
  MLX adapter directory (``<output>/mlx_adapter``).
* The run is recorded in ``<output>/run_history.json`` with
  ``hyperparameters.backend == "mlx"`` (the run is honestly tagged as an MLX
  run, not a CUDA run).

The trainer-wiring contracts (the ``auto``→``mlx`` resolution, the forced-mlx-
on-non-Apple guard, the LoRA-SFT-only gates, the PEFT-alpha→mlx-``scale``
mapping, the run-history tagging, the stdout loss parse) are pinned by the
mlx-MOCKED unit suite in ``tests/test_mlx_backend.py``. THIS file is the
complementary "the bytes actually flow through real ``mlx_lm.lora``" proof —
deliberately the only test in the suite that spins up the MLX toolchain for real.

⚠️ BUILT-BUT-UNVERIFIED on non-Apple rigs ⚠️
--------------------------------------------
``mlx-lm`` is **Apple-Silicon-ONLY** (macOS + arm64) and could NOT be installed
or exercised on the Windows / CUDA rig this test was authored on. The MLX rail
therefore ships in v1.5 as built + unit-tested (mocked); this end-to-end smoke
SKIPS cleanly on every non-Apple host. On an M-series Mac with the ``[mlx]``
extra installed it RUNS and must pass — that real-hardware run is what graduates
the rail from "built" to "verified". Mirror of the FP8 smoke's "experimental,
skips on unsupported hardware" honesty discipline (tests/test_fp8_smoke.py).

Gating (mirrors tests/test_orpo_smoke.py + tests/test_fp8_smoke.py)
------------------------------------------------------------------
* ``@pytest.mark.slow`` AND ``@pytest.mark.integration`` — the fast CI lane
  (``-m "not gpu and not slow and not integration"``) deselects it; it runs in
  the nightly / opt-in lane (``-m "slow or integration"``).
* ``@pytest.mark.skipif(not _APPLE_MLX, ...)`` — skips cleanly, NEVER a silent
  pass, on any host that is not an Apple-Silicon Mac with ``mlx_lm`` importable.
  The skip reason names the exact fix.

Model
-----
``mlx-community/Qwen2.5-0.5B-Instruct-4bit`` — a 0.5B-param instruct model
published in the ``mlx-community`` org in MLX format with a real chat template
(SFT renders the turns through it). Small enough to download + train 2 steps
quickly on an M-series Mac; it is the same model shape the README MLX example
uses. Any small ``mlx-community/*`` instruct checkpoint that ``mlx_lm.lora`` can
load works as a substitute.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import pytest

import backpropagate.feature_flags as feature_flags

# The tiny mlx-community instruct model under test. MLX-format checkpoint with a
# chat template (SFT needs the template to render the prompt/response turns).
_SMOKE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


# Resolve the Apple-Silicon + mlx-toolchain gate ONCE at import time so the skip
# reason is specific. ``_APPLE_MLX`` is the exact conjunction the trainer's
# ``detect_apple_silicon()`` uses: Darwin + arm64 + the ``mlx`` feature flag
# (``mlx_lm`` importable). On THIS Windows/CUDA rig ``platform.system()`` is
# "Windows" so this is False and the test SKIPS — the honest signal, not a fake
# pass. (We read ``platform`` + ``check_feature`` directly rather than calling
# ``detect_apple_silicon()`` so the skip predicate is self-contained and obvious.)
_APPLE_MLX = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and feature_flags.check_feature("mlx")
)

_SKIP_REASON = (
    "MLX smoke requires an Apple-Silicon Mac (macOS + arm64) with mlx-lm "
    "installed: run on Apple Silicon (M-series Mac) with "
    "pip install 'backpropagate[mlx]' — BUILT-BUT-UNVERIFIED on non-Apple rigs "
    "(mlx-lm is Apple-Silicon-ONLY; it cannot install on Windows / Linux / "
    "Intel Macs, so this rig SKIPS rather than fakes a pass)."
)


# Inline SFT rows (ShareGPT-ish chat). MLX SFT trains on chat records rendered
# through the model's chat template — no preference pairs. A handful of short
# rows is enough to take ~2 optimizer steps without the download dominating.
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


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _APPLE_MLX, reason=_SKIP_REASON)
def test_mlx_end_to_end_trains_and_saves_adapter(tmp_path: Path) -> None:
    """Real MLX LoRA SFT run: download → train ~2 steps → verify adapter + history.

    Runs ONLY on an Apple-Silicon Mac with ``mlx_lm`` installed (skips cleanly
    everywhere else). Exercises the genuine ``mlx_lm.lora`` toolchain through the
    subprocess seam — NOT mocked. Asserts the three load-bearing post-conditions
    of the MLX backend:

    1. ``train()`` returns a ``TrainingRun`` with a FINITE ``final_loss``
       (``mlx_lm.lora`` actually ran and the stdout loss parse succeeded).
    2. At least one ``*.safetensors`` adapter file is written under the MLX
       adapter dir (``<output>/mlx_adapter``) — a real, loadable LoRA adapter.
    3. ``run_history.json`` records the run with ``backend == "mlx"`` (the run is
       honestly tagged as an MLX run, not CUDA).
    """
    from backpropagate.checkpoints import RunHistoryManager
    from backpropagate.trainer import Trainer, TrainingRun

    # --- inputs -----------------------------------------------------------
    data_path = tmp_path / "sft.jsonl"
    with open(data_path, "w", encoding="utf-8") as fh:
        for row in _SFT_ROWS:
            fh.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "mlx_output"

    # --- trainer ----------------------------------------------------------
    # backend="mlx" forces the Apple-Silicon rail; mode="lora" + method="sft"
    # is the ONLY MLX-supported combination in v1.5. lora_r=4 + max_seq=128
    # keep the step tiny. use_unsloth is irrelevant on the MLX rail (the toolchain
    # loads the model itself in-subprocess) — left at the default.
    trainer = Trainer(
        model=_SMOKE_MODEL,
        backend="mlx",
        mode="lora",
        method="sft",
        lora_r=4,
        max_seq_length=128,
        output_dir=str(output_dir),
        batch_size=1,
        gradient_accumulation=1,
        report_to="none",
    )

    # The effective rail should have resolved to "mlx" on this Apple host (the
    # skip gate confirmed Darwin + arm64 + mlx_lm) — a cheap sanity check that
    # we're on the MLX path before we spend the download.
    assert trainer.backend == "mlx"
    assert trainer._effective_backend == "mlx", (
        "backend='mlx' on an Apple-Silicon host (the skip gate confirmed "
        "Darwin + arm64 + mlx_lm) but _effective_backend is not 'mlx' — the "
        "backend resolution degraded when it should not have."
    )

    # --- train (REAL) -----------------------------------------------------
    # ~2 optimizer steps via mlx_lm.lora. steps= is the iter cap.
    run = trainer.train(str(data_path), steps=2)

    # 1. Finite final loss from a real mlx_lm.lora run.
    assert isinstance(run, TrainingRun), (
        f"train() should return a TrainingRun; got {type(run)!r}."
    )
    assert run.final_loss is not None
    assert math.isfinite(run.final_loss), (
        f"MLX final_loss must be finite; got {run.final_loss!r}. A non-finite "
        "loss means the mlx_lm.lora step diverged, or the stdout loss parse "
        "matched a garbage token."
    )

    # 2. A real adapter on disk: at least one *.safetensors under the MLX adapter
    # dir. mlx_lm.lora writes its adapter (e.g. adapters.safetensors) into the
    # adapter_path the trainer hands it (<output>/mlx_adapter). The filename has
    # drifted across mlx-lm versions, so we assert on the *.safetensors glob
    # rather than a hard-coded name — the version-stable, load-bearing check.
    adapter_dir = output_dir / "mlx_adapter"
    assert adapter_dir.is_dir(), (
        f"MLX adapter dir {adapter_dir} was not created — mlx_lm.lora did not "
        "write to the adapter_path the trainer passed it."
    )
    safetensors = list(adapter_dir.rglob("*.safetensors"))
    assert safetensors, (
        f"no *.safetensors adapter file under {adapter_dir} — mlx_lm.lora did "
        "not write a LoRA adapter."
    )
    # The adapter file should carry actual tensor bytes, not be a 0-byte stub.
    assert any(p.stat().st_size > 0 for p in safetensors), (
        "every *.safetensors under the MLX adapter dir is empty."
    )

    # --- run history ------------------------------------------------------
    # 3. The run is recorded and honestly tagged backend='mlx'. Run history
    # lives next to the trainer's output_dir (where _train_with_mlx persisted it).
    history = RunHistoryManager(str(output_dir))
    record = history.get_run(run.run_id)
    assert record is not None, (
        f"run_id {run.run_id} not found in run_history.json under {output_dir}."
    )
    hyperparameters = record.get("hyperparameters", {})
    assert hyperparameters.get("backend") == "mlx", (
        "run-history must tag the run backend='mlx'; got "
        f"{hyperparameters.get('backend')!r}."
    )
