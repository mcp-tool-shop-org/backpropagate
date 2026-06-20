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
import random
import re
import string
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
    # C3: deterministic, judge-free task metrics + helpers (CLI binds these).
    "TASK_METRICS",
    "DEFAULT_TASK_METRICS",
    "normalize_squad_text",
    "normalized_exact_match",
    "token_f1",
    "contains_match",
    "regex_match",
    "pass_rate",
    "compute_task_metric",
    "bootstrap_ci_halfwidth",
]

# C3: the underpowered-eval threshold. Below this many scored held-out items a
# task-metric delta is dominated by sampling noise (Card et al. EMNLP 2020 on
# statistical power); eval_gate emits a loud underpowered warning but does not
# block on power alone. ~100 is the research-grounded floor cited in the
# v1.6 design-lock.
UNDERPOWERED_EVAL_N: int = 100

# C3: default noise-band half-width used by eval_gate when a metric has no
# bootstrap CI to compare against. Deliberately small (a 5-point swing) so a
# genuine regression still gates while sub-noise jitter passes.
DEFAULT_METRIC_TOL: float = 0.05


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

    C3 (v1.6): held-out loss is demoted from "the signal" to a non-regression
    floor; the load-bearing signal is the deterministic, judge-free
    ``task_metrics`` map (HELM arXiv:2211.09110, SQuAD EM/F1 arXiv:1606.05250).
    The new fields are ALWAYS present (``task_metrics`` may be empty), defaulted,
    and declared AFTER ``n_prompts`` so existing 6-positional-arg construction
    keeps working.

    Attributes (C3 additions):
        task_metrics: ``{metric_name: mean_score}`` over the references set.
            Always present; empty ``{}`` when no references were supplied.
        eval_n: Number of held-out reference items the task metrics were scored
            over (0 when no task metrics ran). Drives the underpowered warning.
        metric_ci: ``{metric_name: bootstrap_ci_half_width}`` when a CI was
            feasible (>= 2 scored items), else ``None``. The eval gate compares
            a metric delta to this half-width to separate noise from regression.
    """

    run_id: str
    model_name: str
    held_out_loss: float | None
    perplexity: float | None
    generations: list[GenerationSample] = field(default_factory=list)
    n_prompts: int = 0
    # C3 additions (defaulted + declared last so positional construction holds).
    task_metrics: dict[str, float] = field(default_factory=dict)
    eval_n: int = 0
    metric_ci: dict[str, float] | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict (persisted onto the run-history entry)."""
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "held_out_loss": self.held_out_loss,
            "perplexity": self.perplexity,
            "n_prompts": self.n_prompts,
            "generations": [asdict(g) for g in self.generations],
            # C3: task metrics + power + CI (always serialized; CLI/diff read).
            "task_metrics": dict(self.task_metrics),
            "eval_n": self.eval_n,
            "metric_ci": (dict(self.metric_ci) if self.metric_ci is not None else None),
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


# =============================================================================
# C3 — DETERMINISTIC, JUDGE-FREE TASK METRICS
# =============================================================================
#
# Grounded in the v1.6 design-lock research section: loss/perplexity is a WEAK
# proxy (HELM arXiv:2211.09110), so we add deterministic task metrics that
# answer "did the right answer come out?" without an LLM judge (which is
# nondeterministic, biased, and gameable — MT-Bench null-model gaming
# arXiv:2410.07137). SQuAD normalization + token-F1 are the EM/F1 pair from
# arXiv:1606.05250. ROUGE-L/BLEU are deliberately NOT defaults (Goodhart).
#
# Every metric is a pure function ``(prediction: str, references: list[str]) ->
# float in [0, 1]`` taking the MAX over references (any reference matching is a
# match — the standard SQuAD multi-reference rule). ``pass_rate`` is the one
# exception: its "references" are code-test snippets, scored as the fraction
# that pass.

# A token that is *only* an English article is dropped in SQuAD normalization.
_SQUAD_ARTICLES = {"a", "an", "the"}
# Precompiled punctuation-stripping translation table (drop all ASCII punct).
_PUNCT_TABLE = {ord(c): None for c in string.punctuation}


def normalize_squad_text(text: str) -> str:
    """SQuAD answer normalization (arXiv:1606.05250 official scorer).

    Lowercase -> strip ASCII punctuation -> drop the articles ``a``/``an``/``the``
    as WHOLE tokens (never as substrings — ``"theme"`` keeps its ``the``) ->
    collapse runs of whitespace to single spaces. Deterministic and
    locale-independent.
    """
    lowered = str(text).lower()
    # Strip punctuation BEFORE tokenizing so "paris." -> "paris".
    no_punct = lowered.translate(_PUNCT_TABLE)
    tokens = [t for t in no_punct.split() if t and t not in _SQUAD_ARTICLES]
    return " ".join(tokens)


def _as_reference_list(references: Any) -> list[str]:
    """Coerce a reference spec to a list of strings (tolerate a bare string)."""
    if references is None:
        return []
    if isinstance(references, str):
        return [references]
    return [str(r) for r in references]


def normalized_exact_match(prediction: str, references: Any) -> float:
    """1.0 iff the SQuAD-normalized prediction equals ANY normalized reference.

    Closed-form / single-fact answers (the default closed-form metric in the
    design-lock). Returns the MAX over references (standard SQuAD rule).
    """
    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    pred_norm = normalize_squad_text(prediction)
    return 1.0 if any(pred_norm == normalize_squad_text(r) for r in refs) else 0.0


def _f1_against_one(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """SQuAD token-overlap F1 for a single (prediction, reference) pair."""
    # SQuAD convention: if either side is empty, F1 is 1.0 only when BOTH are
    # empty (a correct "no answer"), else 0.0.
    if not pred_tokens or not ref_tokens:
        return 1.0 if (not pred_tokens and not ref_tokens) else 0.0
    # Multiset (bag) intersection — counts repeated tokens correctly.
    from collections import Counter

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def token_f1(prediction: str, references: Any) -> float:
    """SQuAD-style token-overlap F1 (arXiv:1606.05250), MAX over references.

    Short-answer correctness — partial credit for partial token overlap.
    """
    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    pred_tokens = normalize_squad_text(prediction).split()
    return max(
        _f1_against_one(pred_tokens, normalize_squad_text(r).split()) for r in refs
    )


def contains_match(prediction: str, references: Any) -> float:
    """1.0 iff the (lowercased) prediction CONTAINS any (lowercased) reference.

    A format / presence check — "the answer string appears somewhere in the
    output" — without requiring an exact match. NOT SQuAD-normalized (substring
    containment, so punctuation/articles are preserved).
    """
    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    pred_low = str(prediction).lower()
    return 1.0 if any(str(r).lower() in pred_low for r in refs) else 0.0


def regex_match(prediction: str, references: Any) -> float:
    """1.0 iff the prediction matches any reference treated as a regex pattern.

    Format conformance (e.g. ``\\d{4}`` for a 4-digit code). ``re.search`` is
    used (anywhere in the string). A malformed pattern is a USER error, not a
    silent miss, so it raises ``UserInputError(INPUT_VALIDATION_FAILED)``.
    """
    from backpropagate.exceptions import UserInputError

    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    for pattern in refs:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise UserInputError(
                f"Invalid regex reference pattern {pattern!r}: {exc}",
                code="INPUT_VALIDATION_FAILED",
                hint=(
                    "The 'regex' metric treats each reference as a Python "
                    "regular expression. Fix the pattern or switch to the "
                    "'contains' metric for a plain-substring check."
                ),
            ) from exc
        if compiled.search(str(prediction)):
            return 1.0
    return 0.0


def pass_rate(prediction: str, test_snippets: Any) -> float:
    """Best-effort code ``pass@1``-style metric: fraction of test snippets the

    generated code satisfies. The model output ``prediction`` is exec'd once to
    define its symbols, then each test snippet is exec'd against that namespace;
    a snippet "passes" iff it runs without raising (typically an ``assert``).

    Sandboxing is BEST-EFFORT, not a security boundary: builtins are restricted
    to a small safe subset and there is no network/file isolation. Only run this
    on code you would already run locally. A syntax error in the generated code,
    or any snippet raising, scores that snippet 0 rather than crashing the eval.

    Returns ``0.0`` when ``test_snippets`` is empty (nothing proven).
    """
    snippets = _as_reference_list(test_snippets)
    if not snippets:
        return 0.0

    # A restricted builtins map — enough for typical asserts / simple helpers,
    # without obvious foot-guns. NOT a real sandbox (see docstring).
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "divmod": divmod, "enumerate": enumerate, "float": float, "int": int,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "pow": pow, "range": range, "reversed": reversed, "round": round,
        "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "zip": zip, "AssertionError": AssertionError, "Exception": Exception,
    }
    base_globals: dict[str, Any] = {"__builtins__": safe_builtins}

    # Define the model's code once. A syntax/runtime error here means NO snippet
    # can pass -> 0.0 for the whole sample (do not crash the eval).
    try:
        exec(compile(str(prediction), "<generated>", "exec"), base_globals)  # nosec B102 — best-effort, documented
    except Exception as exc:  # noqa: BLE001 — broken generated code => score 0
        logger.debug("pass_rate: generated code failed to define (%s)", exc)
        return 0.0

    passed = 0
    for snippet in snippets:
        snippet_globals = dict(base_globals)
        try:
            exec(compile(str(snippet), "<test>", "exec"), snippet_globals)  # nosec B102 — best-effort, documented
            passed += 1
        except Exception:  # noqa: BLE001  # nosec B112 — a failing test is a 0, not a crash
            continue
    return passed / len(snippets)


# Metric registry: name -> pure scorer. CLI binds against these names.
TASK_METRICS: dict[str, Any] = {
    "normalized_exact_match": normalized_exact_match,
    "token_f1": token_f1,
    "contains": contains_match,
    "regex": regex_match,
    "pass_rate": pass_rate,
}

# The judge-free defaults the design-lock blesses for a closed/short-answer
# held-out set. ROUGE-L/BLEU are intentionally absent — never the default gate.
DEFAULT_TASK_METRICS: list[str] = ["normalized_exact_match", "token_f1"]


def compute_task_metric(metric: str, prediction: str, references: Any) -> float:
    """Dispatch a single metric by name. Unknown metric -> INPUT_ error."""
    from backpropagate.exceptions import UserInputError

    fn = TASK_METRICS.get(metric)
    if fn is None:
        raise UserInputError(
            f"Unknown eval metric {metric!r}.",
            code="INPUT_VALIDATION_FAILED",
            hint=(
                "Pick a deterministic, judge-free metric: "
                f"{', '.join(sorted(TASK_METRICS))}. (ROUGE-L/BLEU are not "
                "wired as gateable metrics by design — see handbook.)"
            ),
        )
    return float(fn(prediction, references))


def bootstrap_ci_halfwidth(
    scores: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> float | None:
    """Bootstrap CI half-width of the MEAN of per-item ``scores`` (in [0, 1]).

    Resamples ``scores`` with replacement ``n_resamples`` times, takes the
    central ``confidence`` interval of the resample means, and returns half its
    width (so a metric delta can be compared to "+/- half-width"). Deterministic
    for a fixed ``seed`` (a LOCAL ``random.Random`` — never touches global RNG
    state). Returns ``None`` when fewer than 2 scores (a CI is meaningless).
    """
    n = len(scores)
    if n < 2:
        return None
    rng = random.Random(seed)  # nosec B311 — bootstrap CI resampling, not crypto
    means: list[float] = []
    for _ in range(n_resamples):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += scores[rng.randrange(n)]
        means.append(resample_sum / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples))
    half_width = (means[hi_idx] - means[lo_idx]) / 2.0
    return max(0.0, half_width)


def _compute_task_metrics(
    generations: list[GenerationSample],
    references: list[dict[str, Any]],
    metrics: list[str],
) -> tuple[dict[str, float], int, dict[str, float] | None]:
    """Score ``generations`` (aligned 1:1 with ``references``) on each metric.

    Returns ``(task_metrics, eval_n, metric_ci)``:
      * ``task_metrics`` — ``{metric: mean_score}`` over the scored items.
      * ``eval_n`` — number of reference items scored.
      * ``metric_ci`` — per-metric bootstrap CI half-width, or ``None`` when a
        CI is infeasible (< 2 items).

    Each reference item is ``{"prompt": str, "reference": str}`` OR
    ``{"prompt": str, "references": [str, ...]}``. A missing reference field is
    a USER error (``INPUT_VALIDATION_FAILED``) — never a silent skip.
    """
    from backpropagate.exceptions import UserInputError

    if not references or not metrics:
        return {}, 0, None

    # Build the per-item reference lists, validating shape up front.
    ref_lists: list[list[str]] = []
    for i, item in enumerate(references):
        if not isinstance(item, dict):
            raise UserInputError(
                f"References item #{i} must be a dict with a 'prompt' and a "
                f"'reference'/'references' key; got {type(item).__name__}.",
                code="INPUT_VALIDATION_FAILED",
                hint=(
                    "Each held-out reference item is "
                    '{"prompt": "...", "reference": "..."} or '
                    '{"prompt": "...", "references": ["...", "..."]}.'
                ),
            )
        if "references" in item:
            refs = _as_reference_list(item["references"])
        elif "reference" in item:
            refs = _as_reference_list(item["reference"])
        else:
            raise UserInputError(
                f"References item #{i} is missing a 'reference'/'references' "
                f"key: {item!r}.",
                code="INPUT_VALIDATION_FAILED",
                hint=(
                    "Add a 'reference' (string) or 'references' (list) field to "
                    "each held-out item so the task metric has a gold answer."
                ),
            )
        ref_lists.append(refs)

    # Per-item, per-metric scores (so we can bootstrap a CI per metric).
    per_metric_scores: dict[str, list[float]] = {m: [] for m in metrics}
    eval_n = min(len(generations), len(ref_lists))
    for idx in range(eval_n):
        prediction = generations[idx].completion
        refs = ref_lists[idx]
        for metric in metrics:
            per_metric_scores[metric].append(
                compute_task_metric(metric, prediction, refs)
            )

    task_metrics: dict[str, float] = {}
    metric_ci: dict[str, float] = {}
    for metric, scores in per_metric_scores.items():
        if not scores:
            continue
        task_metrics[metric] = sum(scores) / len(scores)
        hw = bootstrap_ci_halfwidth(scores)
        if hw is not None:
            metric_ci[metric] = hw

    return task_metrics, eval_n, (metric_ci or None)


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
    heldout_texts: list[str] | None = None,
) -> list[str]:
    """Resolve held-out text per the documented precedence.

    0. ``heldout_texts`` (in-memory list of already-flattened text strings)
       wins outright — this is the seam the SLAO eval gate uses to pass the
       run's reserved last-10% holdout it derived in-process (no on-disk file).
       Empty / blank entries are filtered; an all-empty list raises
       ``INPUT_EVAL_HELDOUT_UNRESOLVED``.
    1. ``heldout`` path next — loaded + flattened to ChatML text.
    2. Else best-effort re-split of the run's ``dataset_info`` (when it names a
       readable on-disk file) with the fixed ``seed``, emitting a loud WARN that
       overlap with the training split is possible.
    3. Neither resolvable -> ``UserInputError(code="INPUT_EVAL_HELDOUT_UNRESOLVED")``.
    """
    from backpropagate.datasets import DatasetLoader
    from backpropagate.exceptions import UserInputError

    # (0) explicit in-memory held-out texts win (the eval-gate hook).
    if heldout_texts is not None:
        cleaned = [str(t).strip() for t in heldout_texts if str(t).strip()]
        if not cleaned:
            raise UserInputError(
                "In-memory held-out set resolved to zero usable text samples.",
                code="INPUT_EVAL_HELDOUT_UNRESOLVED",
                hint=(
                    "The caller passed an explicit held-out text list, but every "
                    "entry was empty after stripping. Pass at least one non-empty "
                    "held-out text."
                ),
            )
        return cleaned

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
    """Unweighted mean of per-sequence cross-entropy over ``texts`` (eval mode, no_grad).

    Returns ``None`` when ``texts`` is empty (caller treats perplexity as
    ``None`` too). Each text is the labels==input_ids language-modelling loss
    that the HF model returns from ``outputs.loss`` (itself a per-token mean over
    that sequence); we average those per-sequence losses with EQUAL weight per
    text — i.e. this is NOT token-weighted, a long sequence counts the same as a
    short one. torch is imported lazily here so module import stays torch-free.
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
            attention_mask = enc.get("attention_mask")
            # Move BOTH tensors to the model's device atomically when the model
            # exposes one (real models do; MagicMocks in tests do not — guard so
            # tests stay CPU). TRAINER-DATA-002: compute both moved tensors into
            # locals FIRST and only commit them together, so a failure mid-move
            # leaves input_ids + attention_mask on the SAME (original) device
            # rather than a half-moved, device-mismatched pair.
            device = getattr(model, "device", None)
            if device is not None:
                try:
                    moved_input_ids = input_ids.to(device)
                    moved_attention_mask = (
                        attention_mask.to(device)
                        if attention_mask is not None
                        else None
                    )
                except Exception:  # nosec B110 — device move is best-effort
                    # Asymmetric failure: keep BOTH on their original device so
                    # the forward pass never sees a split-device pair.
                    pass
                else:
                    input_ids = moved_input_ids
                    attention_mask = moved_attention_mask
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
    heldout_texts: list[str] | None = None,
    prompts: str | None = None,
    n: int = 5,
    seed: int = 0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    metrics: list[str] | None = None,
    references: list[dict[str, Any]] | None = None,
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
        heldout_texts: An already-flattened in-memory list of held-out text
            strings. When provided it takes precedence over ``heldout`` and the
            dataset re-split — the SLAO eval gate uses this to pass the run's
            reserved last-10% holdout it derived in-process (no on-disk file).
        prompts: Path to prompts (one per line OR jsonl ``{"prompt": ...}``); if
            ``None`` use the built-in :data:`DEFAULT_PROMPTS` set.
        n: Number of generations (and prompt truncation length).
        seed: Fixed seed for the re-split + generation determinism.
        max_new_tokens: Generation length cap.
        temperature: Sampling temperature (``<= 0`` -> greedy decoding).
        metrics: C3 — deterministic, judge-free task metrics to compute against
            ``references`` (``normalized_exact_match`` / ``token_f1`` /
            ``contains`` / ``regex`` / ``pass_rate``). When ``references`` is
            given but ``metrics`` is ``None``, defaults to the SQuAD pair
            (:data:`DEFAULT_TASK_METRICS`). Ignored when ``references`` is None.
        references: C3 — a held-out reference set, each item
            ``{"prompt": str, "reference": str}`` or
            ``{"prompt": str, "references": [str, ...]}``. Each prompt is
            generated against (greedy when ``temperature<=0``) and the
            completion scored on every metric; the per-item scores populate
            ``EvalResult.task_metrics`` (mean), ``eval_n``, and ``metric_ci``
            (bootstrap half-width). When ``None`` (default) ``task_metrics`` is
            ``{}`` and behavior is byte-identical to the pre-C3 surface.

    Raises:
        UserInputError: ``INPUT_EVAL_RUN_NOT_FOUND`` (unknown run) /
            ``INPUT_EVAL_HELDOUT_UNRESOLVED`` (no held-out resolvable) /
            ``INPUT_VALIDATION_FAILED`` (bad metric name or reference shape).
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
    held_out_texts = _resolve_heldout_texts(run, heldout, seed, heldout_texts)
    prompt_set = _load_prompts(prompts, n)

    # C3: resolve the task-metric selection + reference prompts up front so a
    # bad metric name / reference shape also fails fast (before the model load).
    metric_names: list[str] = []
    reference_items: list[dict[str, Any]] = []
    if references:
        metric_names = list(metrics) if metrics else list(DEFAULT_TASK_METRICS)
        # Validate every metric name now (cheap), so an unknown metric never
        # surfaces only after the model download + generation.
        for m in metric_names:
            if m not in TASK_METRICS:
                from backpropagate.exceptions import UserInputError as _UIE

                raise _UIE(
                    f"Unknown eval metric {m!r}.",
                    code="INPUT_VALIDATION_FAILED",
                    hint=(
                        "Pick a deterministic, judge-free metric: "
                        f"{', '.join(sorted(TASK_METRICS))}."
                    ),
                )
        reference_items = list(references)
        # Build the reference prompt list, validating shape (a missing prompt is
        # a clear INPUT_ error, not a silent skip). The reference-answer shape is
        # validated inside _compute_task_metrics.
        for i, item in enumerate(reference_items):
            if not (isinstance(item, dict) and item.get("prompt")):
                raise UserInputError(
                    f"References item #{i} must be a dict with a non-empty "
                    f"'prompt' key; got {item!r}.",
                    code="INPUT_VALIDATION_FAILED",
                    hint=(
                        "Each held-out reference item is "
                        '{"prompt": "...", "reference": "..."} or '
                        '{"prompt": "...", "references": ["...", "..."]}.'
                    ),
                )
    reference_prompts = [str(item["prompt"]) for item in reference_items]

    # Load model + tokenizer + adapter (lazy heavy imports inside).
    model, tokenizer = _load_model_and_tokenizer(run)

    # Determine a safe truncation length for held-out scoring from the run's
    # recorded hyperparameters when available; else a conservative default.
    hyperparameters = run.get("hyperparameters") or {}
    try:
        max_length = int(hyperparameters.get("max_seq_length") or 1024)
    except (TypeError, ValueError):
        max_length = 1024

    task_metrics: dict[str, float] = {}
    eval_n = 0
    metric_ci: dict[str, float] | None = None
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
        # C3: generate against the reference prompts (greedy — temperature=0 —
        # so the scored completion is deterministic regardless of the caller's
        # sampling temperature) and score the deterministic task metrics.
        if reference_prompts:
            ref_generations = _generate(
                model,
                tokenizer,
                reference_prompts,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                seed=seed,
            )
            task_metrics, eval_n, metric_ci = _compute_task_metrics(
                ref_generations, reference_items, metric_names
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
        task_metrics=task_metrics,
        eval_n=eval_n,
        metric_ci=metric_ci,
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
        "evaluate_run: run_id=%s held_out_loss=%s perplexity=%s n_prompts=%d "
        "task_metrics=%s eval_n=%d",
        resolved_run_id,
        _fmt(held_out_loss),
        _fmt(perplexity),
        len(prompt_set),
        {k: round(v, 4) for k, v in task_metrics.items()},
        eval_n,
    )
    return result


def diff_evals(a: EvalResult, b: EvalResult) -> EvalDiff:
    """Build a ``cmd_diff_runs``-style row table comparing two evals.

    Each row is ``(metric_name, value_a, value_b)`` with values pre-formatted
    for a two-column terminal table (``None`` renders as ``"n/a"``). The model
    name, held-out loss, perplexity, prompt count, and every C3 task metric
    (the union of both sides' metric keys) are surfaced; a metric present on
    only one side renders ``"n/a"`` for the missing side.
    """
    rows: list[tuple[str, str, str]] = [
        ("model_name", a.model_name or "n/a", b.model_name or "n/a"),
        ("held_out_loss", _fmt(a.held_out_loss), _fmt(b.held_out_loss)),
        ("perplexity", _fmt(a.perplexity), _fmt(b.perplexity)),
        ("n_prompts", str(a.n_prompts), str(b.n_prompts)),
    ]
    # C3: surface task metrics. Union the metric keys (sorted for stable order)
    # so a metric on only one side still shows up (with "n/a" on the other).
    metric_keys = sorted(set(a.task_metrics) | set(b.task_metrics))
    for key in metric_keys:
        rows.append((key, _fmt(a.task_metrics.get(key)), _fmt(b.task_metrics.get(key))))
    if a.eval_n or b.eval_n:
        rows.append(("eval_n", str(a.eval_n), str(b.eval_n)))
    return EvalDiff(run_id_a=a.run_id, run_id_b=b.run_id, rows=rows)


def eval_gate(
    before: EvalResult,
    after: EvalResult,
    *,
    max_regression: float = 0.0,
    gated_metrics: list[str] | None = None,
    metric_tol: float | None = None,
) -> EvalGateDecision:
    """Decide whether ``after`` is acceptable relative to ``before`` (C3 gate).

    This is the seam the SLAO eval-gated merge consumes. C3 upgrades it from a
    loss-only gate to a **conjunction**: accept iff

      1. the held-out loss does NOT regress beyond ``max_regression`` (loss is
         demoted to a non-regression FLOOR — a weak proxy, HELM arXiv:2211.09110),
         AND
      2. EVERY gated task metric does NOT regress beyond its NOISE-BAND tolerance
         — the metric delta is compared to the metric's bootstrap CI half-width
         (``after.metric_ci`` / ``before.metric_ci``) when available, else to
         ``metric_tol`` (default :data:`DEFAULT_METRIC_TOL`). A drop SMALLER than
         the band is sampling noise and does NOT gate (Card et al. EMNLP 2020).

    Loss is a floor: a loss regression rejects even if a metric improved. A real
    metric regression rejects even if loss improved — the reject message frames
    this contrastively (e.g. "loss improved 0.50 BUT exact_match dropped 20pts
    +/- 2 -> real regression"). When a gated metric is absent from either side
    the gate cannot prove non-regression and **rejects** (fail-safe).

    ``regression`` on the returned decision stays the LOSS delta
    (``after_loss - before_loss``; NaN when a loss is missing) for backward
    compatibility with callers reading that float.

    When ``after.eval_n`` is below :data:`UNDERPOWERED_EVAL_N` and metrics are
    gated, a loud underpowered WARNING is logged (the gate still returns a
    decision — it does not block on statistical power alone).

    Args:
        before: Baseline eval (e.g. the pre-merge accumulator).
        after: Candidate eval (e.g. the post-merge model).
        max_regression: Maximum tolerated held-out-loss increase (floor).
        gated_metrics: Task-metric names that must not regress beyond their
            noise band. ``None``/empty -> loss-only gate (pre-C3 behavior).
        metric_tol: Explicit noise-band half-width for ALL gated metrics. When
            ``None``, each metric uses its bootstrap CI half-width if present,
            else :data:`DEFAULT_METRIC_TOL`.

    Returns:
        :class:`EvalGateDecision` whose ``reason`` names the deciding signal.
    """
    before_loss = before.held_out_loss
    after_loss = after.held_out_loss

    # ---- (0) underpowered signal (warn, never block on power alone) --------
    gated = list(gated_metrics or [])
    if gated and 0 < after.eval_n < UNDERPOWERED_EVAL_N:
        logger.warning(
            "eval_gate: UNDERPOWERED — task metrics scored over only "
            "eval_n=%d held-out items (< %d). Metric deltas at this n are "
            "dominated by sampling noise; treat the gate verdict as low-"
            "confidence and enlarge the held-out reference set for a "
            "trustworthy decision (Card et al. EMNLP 2020 on power).",
            after.eval_n,
            UNDERPOWERED_EVAL_N,
        )

    # ---- (1) loss floor ----------------------------------------------------
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
    loss_ok = regression <= max_regression

    if regression < 0:
        loss_phrase = (
            f"loss improved {-regression:.4f} ({before_loss:.4f}->{after_loss:.4f})"
        )
    elif loss_ok:
        loss_phrase = (
            f"loss regression {regression:.4f} within "
            f"max_regression={max_regression:.4f} "
            f"({before_loss:.4f}->{after_loss:.4f})"
        )
    else:
        loss_phrase = (
            f"loss regressed {regression:.4f} exceeding "
            f"max_regression={max_regression:.4f} "
            f"({before_loss:.4f}->{after_loss:.4f})"
        )

    # Loss is a FLOOR: a loss regression beyond tol rejects regardless of metrics.
    if not loss_ok:
        return EvalGateDecision(
            accept=False,
            reason=f"rejected: {loss_phrase} (loss is a non-regression floor).",
            regression=regression,
        )

    # ---- (2) task-metric conjunction (noise-band) --------------------------
    if not gated:
        # Pre-C3 loss-only gate (no metrics to gate on).
        return EvalGateDecision(
            accept=True, reason=f"accepted: {loss_phrase}.", regression=regression
        )

    before_ci = before.metric_ci or {}
    after_ci = after.metric_ci or {}
    metric_notes: list[str] = []
    for metric in gated:
        b_val = before.task_metrics.get(metric)
        a_val = after.task_metrics.get(metric)
        if b_val is None or a_val is None:
            # Cannot prove non-regression for a metric absent on a side.
            return EvalGateDecision(
                accept=False,
                reason=(
                    f"rejected: gated metric {metric!r} is missing on "
                    f"{'before' if b_val is None else 'after'} eval "
                    f"(have before={b_val}, after={a_val}); cannot prove it "
                    "did not regress — rejecting fail-safe. ("
                    f"{loss_phrase}.)"
                ),
                regression=regression,
            )
        delta = a_val - b_val  # positive == improved (metrics are "higher better")
        # Noise band: explicit tol wins; else the larger of the two sides' CI
        # half-widths (the more conservative band); else the default tol.
        if metric_tol is not None:
            band = metric_tol
        else:
            ci_candidates = [
                c for c in (after_ci.get(metric), before_ci.get(metric)) if c is not None
            ]
            band = max(ci_candidates) if ci_candidates else DEFAULT_METRIC_TOL
        drop = -delta  # positive == regressed
        if drop > band:
            # A real (beyond-noise) metric regression. Contrastive message:
            # name the loss improvement (if any) AND the metric drop + band.
            return EvalGateDecision(
                accept=False,
                reason=(
                    f"rejected: {loss_phrase} BUT {metric} dropped "
                    f"{drop:.4f} +/- {band:.4f} -> real regression beyond the "
                    "noise band (metric is the load-bearing signal; loss is "
                    "only a floor)."
                ),
                regression=regression,
            )
        # Within noise (or improved): record a contrastive note for the accept.
        if drop > 0:
            metric_notes.append(
                f"{metric} dropped {drop:.4f} +/- {band:.4f} -> within noise, "
                "not gating"
            )
        else:
            metric_notes.append(f"{metric} improved {-drop:.4f}")

    return EvalGateDecision(
        accept=True,
        reason=(
            f"accepted: {loss_phrase}; "
            + "; ".join(metric_notes)
            + " (conjunction gate: loss floor + task-metric noise band)."
        ),
        regression=regression,
    )
