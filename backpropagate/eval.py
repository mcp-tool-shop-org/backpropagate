"""
Backpropagate — Lightweight post-train evaluation harness (v1.5 T1.1)
=====================================================================

The eval half of the v1.5 dataset-quality + eval loop (``docs/V1_5_BRIEF.md``
§4 "T1.1"). It answers the question a solo finetuner actually asks after a
run: *did this adapter get better or worse?* — without standing up a heavy
benchmark harness.

What it does
------------
- :func:`evaluate_run` resolves a completed run from the on-disk run history,
  re-derives (or accepts) a held-out set, loads the run's base model + LoRA
  adapter, and computes **mean cross-entropy held-out loss + perplexity** plus
  ``n`` sample generations against a fixed prompt set. The result is persisted
  back onto the run-history entry (``extra={"eval": ...}``) so a later
  ``backprop show-run`` / diff can read it without recomputing.
- :func:`diff_evals` produces a ``cmd_diff_runs``-style row table comparing two
  :class:`EvalResult` objects.
- :func:`eval_gate` is the decision seam the v1.5 SLAO eval-gated merge
  (T2.2) consumes: it rejects an ``after`` whose held-out loss regressed past
  ``max_regression``.

torch discipline (load-bearing)
-------------------------------
**All** torch / transformers / peft imports are lazy (inside functions). The
module top level imports only stdlib + dataclasses + typing so
``import backpropagate.eval`` stays cheap and a sibling can re-export these
public names from ``__init__.py`` without pulling torch into the process. The
model load is self-contained (transformers ``AutoModelForCausalLM`` +
``AutoTokenizer`` and ``peft.PeftModel.from_pretrained`` for the adapter) — it
does NOT depend on private ``Trainer`` internals.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationSample",
    "EvalResult",
    "EvalGateDecision",
    "EvalDiff",
    "evaluate_run",
    "diff_evals",
    "eval_gate",
    "DEFAULT_PROMPTS",
]


# A tiny built-in prompt set used when the caller passes no ``prompts`` file.
# Deliberately generic, instruction-shaped, and short so a single tiny eval
# pass over them is cheap on a 16GB consumer GPU. Five entries matches the
# ``n=5`` default so a no-arg ``evaluate_run`` exercises each prompt once.
DEFAULT_PROMPTS: list[str] = [
    "Explain what a neural network is in one sentence.",
    "Write a haiku about the ocean.",
    "List three uses for a paperclip.",
    "Summarize the plot of Cinderella in two sentences.",
    "What is the capital of France?",
]


# =============================================================================
# DATA CLASSES (FIXED PUBLIC API CONTRACT — the CLI agent codes against these)
# =============================================================================

@dataclass
class GenerationSample:
    """A single (prompt, completion) pair produced during evaluation."""

    prompt: str
    completion: str


@dataclass
class EvalResult:
    """The full result of evaluating one run.

    ``held_out_loss`` / ``perplexity`` are ``None`` when no held-out set could
    be scored (e.g. an empty held-out file) — the generations still populate.
    """

    run_id: str
    model_name: str
    held_out_loss: float | None
    perplexity: float | None
    generations: list[GenerationSample] = field(default_factory=list)
    n_prompts: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict (persisted onto the run-history entry)."""
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "held_out_loss": self.held_out_loss,
            "perplexity": self.perplexity,
            "n_prompts": self.n_prompts,
            "generations": [asdict(g) for g in self.generations],
        }


@dataclass
class EvalGateDecision:
    """The accept/reject verdict from :func:`eval_gate`.

    ``regression`` is ``after_loss - before_loss`` (positive == got worse).
    """

    accept: bool
    reason: str
    regression: float  # after_loss - before_loss


@dataclass
class EvalDiff:
    """A ``cmd_diff_runs``-style comparison of two :class:`EvalResult` objects."""

    run_id_a: str
    run_id_b: str
    # (metric_name, value_a, value_b) rows, ready for a two-column table render.
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id_a": self.run_id_a,
            "run_id_b": self.run_id_b,
            # tuples -> lists so the dict round-trips cleanly through JSON.
            "rows": [list(row) for row in self.rows],
        }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _fmt(value: float | None) -> str:
    """Render a metric value for the diff table; ``None`` -> ``"n/a"``."""
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _load_prompts(prompts: str | None, n: int) -> list[str]:
    """Resolve the generation prompt set.

    Precedence: an explicit ``prompts`` path (one prompt per line OR a JSONL of
    ``{"prompt": ...}`` objects) wins; otherwise the built-in
    :data:`DEFAULT_PROMPTS` set is used. The list is truncated to ``n``.

    Raises:
        UserInputError: when ``prompts`` is given but the file does not exist
            or yields zero usable prompts.
    """
    from backpropagate.exceptions import UserInputError

    if prompts is None:
        chosen = list(DEFAULT_PROMPTS)
    else:
        path = Path(prompts)
        if not path.exists():
            raise UserInputError(
                f"Prompts file not found: {prompts}",
                code="INPUT_VALIDATION_FAILED",
                hint=(
                    "Pass --prompts pointing at a readable file (one prompt "
                    "per line, or a JSONL of {\"prompt\": ...} objects), or "
                    "omit --prompts to use the built-in 5-prompt set."
                ),
            )
        chosen = []
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # Tolerate both raw lines and JSONL {"prompt": ...} objects.
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        chosen.append(line)
                        continue
                    if isinstance(obj, dict) and obj.get("prompt"):
                        chosen.append(str(obj["prompt"]))
                    else:
                        # A JSON object without a prompt key isn't a usable
                        # prompt; skip rather than feed the model "{...}".
                        continue
                else:
                    chosen.append(line)
        if not chosen:
            raise UserInputError(
                f"Prompts file is empty or has no usable prompts: {prompts}",
                code="INPUT_VALIDATION_FAILED",
                hint=(
                    "Each non-empty line is treated as a prompt (or a JSONL "
                    "{\"prompt\": ...} object). Add at least one prompt, or "
                    "omit --prompts for the built-in set."
                ),
            )

    if n > 0:
        chosen = chosen[:n]
    return chosen


def _heldout_texts_from_path(heldout: str) -> list[str]:
    """Load a held-out set from ``heldout`` and flatten to ChatML text strings.

    Reuses :class:`backpropagate.datasets.DatasetLoader` so every supported
    on-disk format (JSONL / ShareGPT / Alpaca / OpenAI / raw text) is accepted,
    then renders each sample to a single ChatML string for tokenization.

    Raises:
        UserInputError: when the file resolves to zero usable text samples.
    """
    from backpropagate.datasets import DatasetLoader
    from backpropagate.exceptions import UserInputError

    loader = DatasetLoader(heldout, validate=False)
    texts = _loader_to_texts(loader)
    if not texts:
        raise UserInputError(
            f"Held-out file produced no usable samples: {heldout}",
            code="INPUT_EVAL_HELDOUT_UNRESOLVED",
            hint=(
                "Confirm the held-out file is one of the supported formats "
                "(JSONL / ShareGPT / Alpaca / OpenAI / raw text) and contains "
                "at least one non-empty example."
            ),
        )
    return texts


def _loader_to_texts(loader: Any) -> list[str]:
    """Flatten a ``DatasetLoader`` to a list of non-empty ChatML text strings."""
    chatml = loader.to_chatml()
    texts: list[str] = []
    for row in chatml:
        if isinstance(row, dict):
            text = str(row.get("text", "")).strip()
        else:
            text = str(row).strip()
        if text:
            texts.append(text)
    return texts


def _resolve_heldout_texts(
    run: dict[str, Any],
    heldout: str | None,
    seed: int,
) -> list[str]:
    """Resolve held-out text per the documented precedence.

    1. ``heldout`` path wins — loaded + flattened to ChatML text.
    2. Else best-effort re-split of the run's ``dataset_info`` (when it names a
       readable on-disk file) with the fixed ``seed``, emitting a loud WARN that
       overlap with the training split is possible.
    3. Neither resolvable -> ``UserInputError(code="INPUT_EVAL_HELDOUT_UNRESOLVED")``.
    """
    from backpropagate.datasets import DatasetLoader
    from backpropagate.exceptions import UserInputError

    # (1) explicit held-out path wins.
    if heldout is not None:
        return _heldout_texts_from_path(heldout)

    # (2) best-effort re-split of the run's recorded dataset.
    dataset_info = run.get("dataset_info")
    if isinstance(dataset_info, str) and dataset_info and Path(dataset_info).exists():
        logger.warning(
            "evaluate_run: no --heldout provided; re-splitting the run's "
            "recorded dataset %r with seed=%d to derive a held-out set. "
            "OVERLAP WITH THE TRAINING SPLIT IS POSSIBLE — this re-split is "
            "only as faithful as the original train was deterministic. For a "
            "trustworthy held-out loss, pass --heldout pointing at a set the "
            "run never trained on.",
            dataset_info,
            seed,
        )
        loader = DatasetLoader(dataset_info, validate=False)
        _train_loader, test_loader = loader.split(train_ratio=0.9, seed=seed)
        texts = _loader_to_texts(test_loader)
        if texts:
            return texts
        # The dataset existed but the held-out slice flattened to nothing
        # (e.g. a 1-sample dataset where the test slice is empty). Fall through
        # to the unresolved error so the caller gets an actionable message.
        logger.warning(
            "evaluate_run: re-split of %r produced an empty held-out slice; "
            "treating held-out as unresolved.",
            dataset_info,
        )

    # (3) neither path worked.
    raise UserInputError(
        "Could not resolve a held-out set for this run: no --heldout was "
        f"provided and the run's recorded dataset_info ({dataset_info!r}) is "
        "not a readable on-disk file to re-split.",
        code="INPUT_EVAL_HELDOUT_UNRESOLVED",
        hint=(
            "Pass --heldout pointing at a held-out jsonl the run did not train "
            "on. (When dataset_info names an in-memory object or a path that "
            "no longer exists, backpropagate cannot re-derive a held-out "
            "split.)"
        ),
    )


def _compute_held_out_loss(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
) -> float | None:
    """Mean per-token cross-entropy over ``texts`` (eval mode, no_grad).

    Returns ``None`` when ``texts`` is empty (caller treats perplexity as
    ``None`` too). Each text is the labels==input_ids language-modelling loss
    that the HF model returns from ``outputs.loss``; we average the per-example
    losses. torch is imported lazily here so module import stays torch-free.
    """
    import torch  # lazy — keeps module import cheap + torch-free

    if not texts:
        return None

    model.eval()
    total = 0.0
    counted = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"]
            # Move to the model's device when the model exposes one (real
            # models do; MagicMocks in tests do not — guard so tests stay CPU).
            device = getattr(model, "device", None)
            if device is not None:
                try:
                    input_ids = input_ids.to(device)
                    if "attention_mask" in enc:
                        enc["attention_mask"] = enc["attention_mask"].to(device)
                except Exception:  # nosec B110 — device move is best-effort
                    pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=enc.get("attention_mask"),
                labels=input_ids,
            )
            loss = outputs.loss
            loss_value = float(loss.item())
            if math.isfinite(loss_value):
                total += loss_value
                counted += 1
            else:
                logger.warning(
                    "evaluate_run: non-finite held-out loss on one example "
                    "(NaN/inf); skipping it from the mean."
                )

    if counted == 0:
        return None
    return total / counted


def _generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> list[GenerationSample]:
    """Generate one completion per prompt with a fixed seed/temperature.

    The decoded completion strips the echoed prompt prefix when present so the
    sample holds only the model's continuation. torch is imported lazily.
    """
    import torch  # lazy

    model.eval()
    samples: list[GenerationSample] = []
    do_sample = temperature is not None and temperature > 0.0
    for prompt in prompts:
        # Fixed seed per prompt so a given (run, prompt) pair is reproducible
        # across eval invocations (deterministic generations for the contract).
        try:
            torch.manual_seed(seed)
        except Exception:  # nosec B110 — manual_seed is best-effort under mocks
            pass

        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        device = getattr(model, "device", None)
        if device is not None:
            try:
                input_ids = input_ids.to(device)
            except Exception:  # nosec B110 — best-effort device move
                pass

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

        # ``output_ids`` is [batch, seq]; decode the first row. Strip the
        # prompt prefix length so we keep only the continuation.
        prompt_len = int(input_ids.shape[-1])
        full_row = output_ids[0]
        completion_ids = full_row[prompt_len:]
        completion = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        samples.append(GenerationSample(prompt=prompt, completion=completion))

    return samples


def _load_model_and_tokenizer(run: dict[str, Any]) -> tuple[Any, Any]:
    """Self-contained base-model + tokenizer + LoRA-adapter load.

    Uses transformers ``AutoModelForCausalLM`` / ``AutoTokenizer`` for the base
    model named by ``run['model_name']`` and ``peft.PeftModel.from_pretrained``
    to attach the adapter at the run's ``checkpoint_path``. Deliberately does
    NOT import or depend on private ``Trainer`` methods (trainer.py is being
    heavily edited in a later wave). All heavy imports are lazy.

    Raises:
        TrainingError: wrapping any underlying load failure
            (code ``RUNTIME_EVAL_FAILED``).
    """
    from backpropagate.exceptions import TrainingError

    model_name = str(run.get("model_name") or "")
    checkpoint_path = run.get("checkpoint_path")

    try:
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # A missing pad token breaks batched tokenization on many base models;
        # fall back to EOS, which is the conventional eval-time choice.
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, "float16", None),
        )

        model: Any = base_model
        # Attach the LoRA adapter when the run recorded a checkpoint path that
        # exists on disk. A run with no adapter dir (e.g. a base-model eval)
        # still evaluates against the bare base model.
        if checkpoint_path and Path(checkpoint_path).exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            logger.warning(
                "evaluate_run: run %r has no adapter directory on disk at %r; "
                "evaluating the bare base model. (Held-out loss / generations "
                "reflect the base model, not a fine-tuned adapter.)",
                run.get("run_id"),
                checkpoint_path,
            )
        return model, tokenizer
    except Exception as exc:
        raise TrainingError(
            f"Failed to load model/adapter for evaluation of run "
            f"{run.get('run_id')!r} (model={model_name!r}, "
            f"checkpoint={checkpoint_path!r}): {exc}",
            code="RUNTIME_EVAL_FAILED",
            cause=exc,
        ) from exc


# =============================================================================
# PUBLIC API
# =============================================================================

def evaluate_run(
    run_id: str,
    *,
    output_dir: str,
    heldout: str | None = None,
    prompts: str | None = None,
    n: int = 5,
    seed: int = 0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> EvalResult:
    """Evaluate a completed run: held-out loss + perplexity + N generations.

    Resolves ``run_id`` against the on-disk run history under ``output_dir``
    (supports the same unambiguous-prefix match as ``backprop show-run``),
    derives a held-out set (``--heldout`` wins; else best-effort re-split of the
    run's ``dataset_info`` with a loud WARN), loads the run's base model + LoRA
    adapter, computes mean cross-entropy held-out loss + perplexity in eval /
    ``no_grad`` mode, and generates ``n`` completions against the prompt set
    with the fixed ``seed`` / ``temperature`` / ``max_new_tokens``. The result
    is persisted back onto the run-history entry (``extra={"eval": ...}``,
    best-effort) so downstream readers don't recompute.

    Args:
        run_id: The run to evaluate (exact or unambiguous prefix).
        output_dir: Directory holding ``run_history.json``.
        heldout: Path to a held-out jsonl; if ``None``, re-split the run's
            dataset with a loud WARN that overlap is possible.
        prompts: Path to prompts (one per line OR jsonl ``{"prompt": ...}``); if
            ``None`` use the built-in :data:`DEFAULT_PROMPTS` set.
        n: Number of generations (and prompt truncation length).
        seed: Fixed seed for the re-split + generation determinism.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature (``<= 0`` -> greedy decoding).

    Raises:
        UserInputError: ``INPUT_EVAL_RUN_NOT_FOUND`` (unknown run) /
            ``INPUT_EVAL_HELDOUT_UNRESOLVED`` (no held-out resolvable).
        TrainingError: ``RUNTIME_EVAL_FAILED`` (model load / generation crash).

    Returns:
        The populated :class:`EvalResult`.
    """
    from backpropagate.checkpoints import RunHistoryManager
    from backpropagate.exceptions import TrainingError, UserInputError

    history = RunHistoryManager(output_dir)
    run = history.get_run(run_id)
    if run is None:
        raise UserInputError(
            f"No run found for run_id={run_id!r} in {output_dir!r}.",
            code="INPUT_EVAL_RUN_NOT_FOUND",
            hint=(
                "Run `backprop runs` to list available run_ids in this "
                "output_dir. If the run was trained under a different "
                "--output, pass that directory."
            ),
        )

    # Use the run's canonical (full) run_id for persistence + the result, even
    # when the caller passed a prefix.
    resolved_run_id = str(run.get("run_id") or run_id)
    model_name = str(run.get("model_name") or "")

    # Resolve inputs BEFORE the heavy model load so a bad --heldout / --prompts
    # fails fast (and cheap) with a UserInputError rather than after a download.
    held_out_texts = _resolve_heldout_texts(run, heldout, seed)
    prompt_set = _load_prompts(prompts, n)

    # Load model + tokenizer + adapter (lazy heavy imports inside).
    model, tokenizer = _load_model_and_tokenizer(run)

    # Determine a safe truncation length for held-out scoring from the run's
    # recorded hyperparameters when available; else a conservative default.
    hyperparameters = run.get("hyperparameters") or {}
    try:
        max_length = int(hyperparameters.get("max_seq_length") or 1024)
    except (TypeError, ValueError):
        max_length = 1024

    try:
        held_out_loss = _compute_held_out_loss(
            model, tokenizer, held_out_texts, max_length=max_length
        )
        generations = _generate(
            model,
            tokenizer,
            prompt_set,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
    except UserInputError:
        # Input-shaped problems already carry their own stable code; never
        # mask them as a runtime failure.
        raise
    except Exception as exc:
        raise TrainingError(
            f"Evaluation of run {resolved_run_id!r} failed during "
            f"loss/generation: {exc}",
            code="RUNTIME_EVAL_FAILED",
            cause=exc,
        ) from exc

    perplexity: float | None = None
    if held_out_loss is not None and math.isfinite(held_out_loss):
        try:
            perplexity = math.exp(held_out_loss)
        except OverflowError:
            # Astronomically high loss -> perplexity overflows float; report
            # infinity rather than crashing the eval.
            perplexity = math.inf

    result = EvalResult(
        run_id=resolved_run_id,
        model_name=model_name,
        held_out_loss=held_out_loss,
        perplexity=perplexity,
        generations=generations,
        n_prompts=len(prompt_set),
    )

    # Persist the eval back onto the run-history entry (best-effort: a history
    # write failure must never sink the eval the operator just paid for).
    try:
        history.record_run_completed(
            resolved_run_id, extra={"eval": result.to_dict()}
        )
    except Exception as hist_err:  # nosec B110 — persistence is best-effort
        logger.warning(
            "evaluate_run: failed to persist eval result onto run history "
            "for run_id=%s (%s: %s); the returned EvalResult is unaffected.",
            resolved_run_id,
            type(hist_err).__name__,
            hist_err,
        )

    logger.info(
        "evaluate_run: run_id=%s held_out_loss=%s perplexity=%s n_prompts=%d",
        resolved_run_id,
        _fmt(held_out_loss),
        _fmt(perplexity),
        len(prompt_set),
    )
    return result


def diff_evals(a: EvalResult, b: EvalResult) -> EvalDiff:
    """Build a ``cmd_diff_runs``-style row table comparing two evals.

    Each row is ``(metric_name, value_a, value_b)`` with values pre-formatted
    for a two-column terminal table (``None`` renders as ``"n/a"``). The model
    name, held-out loss, perplexity, and prompt count are surfaced.
    """
    rows: list[tuple[str, str, str]] = [
        ("model_name", a.model_name or "n/a", b.model_name or "n/a"),
        ("held_out_loss", _fmt(a.held_out_loss), _fmt(b.held_out_loss)),
        ("perplexity", _fmt(a.perplexity), _fmt(b.perplexity)),
        ("n_prompts", str(a.n_prompts), str(b.n_prompts)),
    ]
    return EvalDiff(run_id_a=a.run_id, run_id_b=b.run_id, rows=rows)


def eval_gate(
    before: EvalResult,
    after: EvalResult,
    *,
    max_regression: float = 0.0,
) -> EvalGateDecision:
    """Decide whether ``after`` is acceptable relative to ``before``.

    This is the seam the v1.5 SLAO eval-gated merge (T2.2) consumes: a merge is
    rejected when it regresses the held-out loss past ``max_regression``.

    ``regression = after.held_out_loss - before.held_out_loss`` (positive ==
    worse). ``accept`` is ``regression <= max_regression``. When either side is
    missing a held-out loss the gate cannot make a quantitative call and
    **rejects** (fail-safe — an ungated merge could silently regress).

    Args:
        before: Baseline eval (e.g. the pre-merge accumulator).
        after: Candidate eval (e.g. the post-merge model).
        max_regression: Maximum tolerated loss increase. ``0.0`` (default)
            means "must not get worse"; a positive value tolerates that much
            absolute regression.

    Returns:
        :class:`EvalGateDecision` whose ``reason`` names the regression.
    """
    before_loss = before.held_out_loss
    after_loss = after.held_out_loss

    if before_loss is None or after_loss is None:
        # Fail-safe: without both losses we cannot prove non-regression. NaN
        # regression signals "indeterminate" to any caller inspecting the float.
        return EvalGateDecision(
            accept=False,
            reason=(
                "Cannot gate: held-out loss missing on "
                f"{'before' if before_loss is None else 'after'} eval "
                "(re-run evaluate_run with a resolvable --heldout). Rejecting "
                "to avoid an unverified merge."
            ),
            regression=float("nan"),
        )

    regression = after_loss - before_loss
    accept = regression <= max_regression

    if accept:
        if regression < 0:
            verdict = (
                f"accepted: held-out loss improved by {-regression:.4f} "
                f"({before_loss:.4f} -> {after_loss:.4f})"
            )
        else:
            verdict = (
                f"accepted: held-out loss regression {regression:.4f} is "
                f"within max_regression={max_regression:.4f} "
                f"({before_loss:.4f} -> {after_loss:.4f})"
            )
    else:
        verdict = (
            f"rejected: held-out loss regressed by {regression:.4f}, exceeding "
            f"max_regression={max_regression:.4f} "
            f"({before_loss:.4f} -> {after_loss:.4f})"
        )

    return EvalGateDecision(accept=accept, reason=verdict, regression=regression)
