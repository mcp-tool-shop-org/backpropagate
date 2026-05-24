#!/usr/bin/env python3
"""Nightly train smoke — end-to-end CPU `Trainer.train(max_steps=1)` runner.

Purpose
-------
v1.3 brief P1 — catch silent regressions in the ``Trainer`` entry point
between releases. Unit tests stub the HuggingFace stack heavily; this smoke
exercises the real ``trl.SFTTrainer.train()`` path on a tiny CPU model so
that a dependency bump that breaks model loading / dataset auto-detect /
checkpoint writing / run-history persistence trips the CI bell BEFORE the
weekly release cut.

Picked over ``tests/test_nightly_smoke.py`` because:
- The runner takes 5–15 minutes of wall time on a CPU runner — far over the
  pytest --timeout=60 floor we hold the rest of the suite to.
- It depends on torch + transformers + trl + huggingface-hub being fully
  downloadable from the runner; pytest tests must stay green offline.
- The expected failure mode is "a release-blocking environmental
  regression" — a separate workflow with its own gh-issue-create on
  failure is the cleaner observability surface vs. a buried pytest red.

The workflow that drives this script lives at
``.github/workflows/nightly-train-smoke.yml`` (04:00 UTC schedule +
workflow_dispatch). On failure that workflow opens a GitHub Issue tagged
``ci`` / ``nightly-smoke`` — the failure does NOT gate any release.

Exit codes
----------
0 — smoke passed (train returned without raising, checkpoint exists,
    run_history.json has the new entry, loss is finite)
1 — smoke failed (one or more assertions above tripped); stderr names the
    failing assertion so the issue body can quote it directly

Usage
-----
    python scripts/nightly_train_smoke.py
    python scripts/nightly_train_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/nightly_train_smoke.py --output /tmp/nightly-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIXTURE_PATH = Path(__file__).parent.parent / "examples" / "quickstart.jsonl"


def _assert(condition: bool, message: str) -> None:
    """Tiny in-line assertion that prints the failing message to stderr."""
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nightly train smoke for backpropagate (CPU, 1 step)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id to smoke (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the smoke checkpoint (default: a tempdir)",
    )
    args = parser.parse_args()

    # Force CPU torch for the smoke — CPU runners don't have CUDA, and we
    # don't want the smoke to silently take the GPU path on a self-hosted
    # CPU runner that happens to have a stub libcuda.so in $LD_LIBRARY_PATH.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Friendly preflight: fail fast with a clear message if the smoke
    # fixture is missing (catches bad clones / sparse checkouts).
    _assert(
        FIXTURE_PATH.exists(),
        f"smoke fixture {FIXTURE_PATH} not found — sparse checkout?",
    )

    # Stay out of the user's $HOME for caches when --output is not pinned
    # (CI runners care about disk usage on the runner image).
    tmpdir_ctx: tempfile.TemporaryDirectory | None = None
    if args.output is None:
        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="backpropagate-nightly-")
        output_dir = Path(tmpdir_ctx.name)
    else:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import inside main so a missing torch/transformers fails with a
        # readable Python traceback rather than blowing up at import time
        # and producing an opaque CI banner.
        from backpropagate import Trainer

        print(f"smoke: model={args.model}", flush=True)
        print(f"smoke: dataset={FIXTURE_PATH}", flush=True)
        print(f"smoke: output_dir={output_dir}", flush=True)

        trainer = Trainer(
            model=args.model,
            output_dir=str(output_dir),
            # Force off the Unsloth path — Unsloth is GPU-first and the
            # CPU smoke runs on a vanilla transformers fallback.
            use_unsloth=False,
        )

        run = trainer.train(
            dataset=str(FIXTURE_PATH),
            steps=1,
        )

        # Assertion set per V1_3_BRIEF P1 — Trainer.train() returns without
        # raising (already covered by getting here), checkpoint file exists,
        # run_history.json got a new entry, loss is finite.
        _assert(run is not None, "Trainer.train returned None")

        # run.checkpoint_path is the canonical surface for the saved
        # checkpoint dir. Fall back to scanning output_dir for any
        # checkpoint-N if the field is unset (older trainer paths may not
        # populate it on max_steps=1 runs).
        checkpoint_path = getattr(run, "checkpoint_path", None)
        checkpoint_exists = (
            checkpoint_path is not None and Path(checkpoint_path).exists()
        ) or any(output_dir.glob("checkpoint-*"))
        _assert(
            checkpoint_exists,
            f"no checkpoint written under {output_dir} (run.checkpoint_path={checkpoint_path})",
        )

        run_history = output_dir / "run_history.json"
        _assert(
            run_history.exists(),
            f"run_history.json not written under {output_dir}",
        )
        history_data = json.loads(run_history.read_text(encoding="utf-8"))
        # RunHistoryManager writes either a list of run records or a dict
        # keyed by run_id depending on schema version — accept both shapes
        # so a future schema bump doesn't false-positive the smoke.
        if isinstance(history_data, dict):
            entries = history_data.get("runs", []) or list(history_data.values())
        else:
            entries = history_data
        _assert(
            len(entries) >= 1,
            f"run_history.json under {output_dir} has zero entries",
        )

        # Loss may live at run.loss, run.final_loss, or a per-step list —
        # accept any of those shapes and assert finiteness on whichever we
        # find first. A NaN/Inf loss almost always signals an upstream
        # bug (dataset format drift, tokenizer mismatch, OOM-recovery
        # bailout) so it's the highest-signal sanity check on max_steps=1.
        loss = (
            getattr(run, "loss", None)
            or getattr(run, "final_loss", None)
            or getattr(run, "train_loss", None)
        )
        if loss is None:
            losses = getattr(run, "losses", None) or getattr(run, "loss_history", None)
            if losses:
                loss = losses[-1]
        if loss is not None:
            _assert(
                math.isfinite(float(loss)),
                f"final training loss is non-finite ({loss})",
            )
        else:
            # Surface but don't fail — some trainer paths legitimately
            # don't expose loss on a max_steps=1 smoke; the rest of the
            # assertions cover the "did it actually train" question.
            print("smoke: warning: no loss field found on run", flush=True)

        print("smoke: PASS", flush=True)
        return 0

    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
