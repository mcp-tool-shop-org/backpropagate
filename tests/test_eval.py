"""Tests for the v1.5 post-train eval harness (``backpropagate.eval``).

torch is MOCKED throughout — these are CPU-only, no real model download. Two
patterns are used (mirroring tests/test_wave6b_features.py + test_trainer.py):

1. For ``evaluate_run`` orchestration, the heavy seam
   ``backpropagate.eval._load_model_and_tokenizer`` is patched to return
   MagicMock model/tokenizer, and the run-history resolution is patched on
   ``RunHistoryManager.get_run`` / ``record_run_completed``.
2. For the loss/generation internals, a lightweight fake ``torch`` is injected
   via ``patch.dict(sys.modules, {"torch": _FakeTorch()})`` so the real
   ``_compute_held_out_loss`` / ``_generate`` code paths run deterministically
   without CUDA or a model.

A subprocess guard asserts that executing ``eval.py``'s module body in
isolation does NOT pull torch/transformers/peft into ``sys.modules`` — the
load-bearing torch-laziness contract for this module.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import backpropagate.eval as ev
from backpropagate.eval import (
    DEFAULT_PROMPTS,
    EvalDiff,
    EvalGateDecision,
    EvalResult,
    GenerationSample,
    diff_evals,
    eval_gate,
    evaluate_run,
)
from backpropagate.exceptions import TrainingError, UserInputError

# =============================================================================
# torch-laziness contract
# =============================================================================

def test_eval_module_body_imports_no_heavy_deps():
    """Executing ``eval.py``'s module body must not pull torch/transformers/peft.

    A sibling agent re-exports these names from ``backpropagate/__init__.py``;
    if eval.py's *own* module body imported torch at the top level, that cheap
    re-export would become a multi-second, CUDA-touching import. The package
    ``__init__`` separately imports ``.trainer`` (which is eagerly torch-y), so
    an in-process ``"torch" not in sys.modules`` check after
    ``import backpropagate.eval`` can never pass — importing any submodule runs
    the package init first. So we load eval.py as an ISOLATED top-level module
    (bypassing the package init) in a clean subprocess and assert the heavy deps
    stayed out. This isolates the contract to *this module's* import surface.
    """
    eval_path = Path(ev.__file__)
    script = textwrap.dedent(
        f"""
        import sys, importlib.util
        spec = importlib.util.spec_from_file_location(
            "be_eval_isolated", {str(eval_path)!r}
        )
        m = importlib.util.module_from_spec(spec)
        # Register BEFORE exec so @dataclass introspection resolves the module.
        sys.modules["be_eval_isolated"] = m
        spec.loader.exec_module(m)
        heavy = [d for d in ("torch", "transformers", "peft") if d in sys.modules]
        assert not heavy, f"eval.py module body pulled heavy deps: {{heavy}}"
        # Sanity: the public API actually loaded.
        assert hasattr(m, "evaluate_run")
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "eval.py is not torch-lazy. stdout="
        f"{proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "OK" in proc.stdout


# =============================================================================
# Fake torch + model/tokenizer scaffolding (no real CUDA / model)
# =============================================================================

class _FakeTensor:
    """Minimal stand-in for a torch tensor used by the eval code paths."""

    def __init__(self, rows):
        # rows: list[list[int]] (2D) — [batch, seq]
        self._rows = rows

    @property
    def shape(self):
        n_rows = len(self._rows)
        n_cols = len(self._rows[0]) if n_rows else 0
        return (n_rows, n_cols)

    def to(self, _device):  # device move is a no-op in the fake
        return self

    def __getitem__(self, idx):
        # output_ids[0] -> the first row (a 1D "tensor" we can slice).
        return _FakeRow(self._rows[idx])


class _FakeRow:
    """A 1D row supporting prefix-strip slicing (``row[prompt_len:]``)."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, sl):
        return _FakeRow(self._values[sl])

    @property
    def values(self):
        return self._values


class _FakeLoss:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeTorch:
    """A tiny fake ``torch`` module exposing only what eval.py touches."""

    float16 = "float16"

    def __init__(self):
        self.manual_seed_calls = []

    def manual_seed(self, seed):
        self.manual_seed_calls.append(seed)

    @contextmanager
    def no_grad(self):
        yield


def _make_tokenizer():
    """Tokenizer mock: callable -> dict with a _FakeTensor input_ids."""
    tok = MagicMock()

    def _tokenize(text, **kwargs):
        # Encode by word count so different texts/prompts get different lengths.
        n = max(1, len(str(text).split()))
        return {
            "input_ids": _FakeTensor([list(range(n))]),
            "attention_mask": _FakeTensor([[1] * n]),
        }

    tok.side_effect = _tokenize
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    tok.decode = MagicMock(return_value="generated completion text")
    return tok


def _make_model(loss_value: float = 0.5):
    """Model mock: callable returns ``.loss``; ``.generate`` returns ids."""
    model = MagicMock()
    # No ``device`` attribute path: configure as None so the device move is
    # skipped (keeps the fake CPU-only).
    model.device = None
    model.eval = MagicMock()

    def _forward(**kwargs):
        return SimpleNamespace(loss=_FakeLoss(loss_value))

    model.side_effect = _forward
    # generate -> a [1, seq] _FakeTensor longer than the prompt so the
    # prefix-strip yields a non-empty completion slice.
    model.generate = MagicMock(
        return_value=_FakeTensor([list(range(64))])
    )
    return model


# =============================================================================
# evaluate_run — happy path with mocked model load + history
# =============================================================================

def _run_entry(**overrides):
    base = {
        "run_id": "abc123def456",
        "model_name": "unsloth/Qwen2.5-3B",
        "dataset_info": "InMemoryDataset",  # non-path => not re-splittable
        "dataset_hash": "deadbeefcafef00d",
        "checkpoint_path": None,
        "hyperparameters": {"max_seq_length": 256},
    }
    base.update(overrides)
    return base


def test_evaluate_run_happy_path(tmp_path):
    """evaluate_run returns a populated EvalResult for a fixed seed.

    Mocks the model load + history resolution; injects a fake torch so the
    real loss/generation code runs deterministically.
    """
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "content": "hello there friend"}]}\n'
        '{"messages": [{"role": "user", "content": "bye"}, '
        '{"role": "assistant", "content": "goodbye now"}]}\n',
        encoding="utf-8",
    )

    model = _make_model(loss_value=0.5)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry()
    history.record_run_completed.return_value = {}

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        result = evaluate_run(
            "abc123",  # prefix match
            output_dir=str(tmp_path),
            heldout=str(heldout),
            n=3,
            seed=0,
        )

    assert isinstance(result, EvalResult)
    # Resolved to the canonical full run_id, not the prefix the caller passed.
    assert result.run_id == "abc123def456"
    assert result.model_name == "unsloth/Qwen2.5-3B"
    # mean CE of constant 0.5 over 2 held-out examples == 0.5
    assert result.held_out_loss == pytest.approx(0.5)
    # perplexity == exp(0.5)
    assert result.perplexity == pytest.approx(2.718281828 ** 0.5, rel=1e-4)
    # n generations == n prompts (DEFAULT_PROMPTS truncated to n=3)
    assert result.n_prompts == 3
    assert len(result.generations) == 3
    assert all(isinstance(g, GenerationSample) for g in result.generations)


def test_evaluate_run_persists_eval_via_record_run_completed(tmp_path):
    """After computing, evaluate_run persists via record_run_completed(extra=...)."""
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a longer answer here"}]}\n',
        encoding="utf-8",
    )

    model = _make_model(loss_value=1.0)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry()

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        result = evaluate_run(
            "abc123def456",
            output_dir=str(tmp_path),
            heldout=str(heldout),
            n=2,
        )

    history.record_run_completed.assert_called_once()
    args, kwargs = history.record_run_completed.call_args
    # Positional run_id is the canonical full id.
    assert args[0] == "abc123def456"
    # The eval dict is merged via the ``extra`` kwarg.
    assert "extra" in kwargs
    assert "eval" in kwargs["extra"]
    persisted = kwargs["extra"]["eval"]
    assert persisted["run_id"] == "abc123def456"
    assert persisted["held_out_loss"] == pytest.approx(1.0)
    assert persisted == result.to_dict()


def test_evaluate_run_persistence_failure_is_swallowed(tmp_path):
    """A history write failure must not sink the returned EvalResult."""
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "an answer"}]}\n',
        encoding="utf-8",
    )

    model = _make_model(loss_value=0.25)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry()
    history.record_run_completed.side_effect = RuntimeError("disk full")

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        result = evaluate_run(
            "abc123def456",
            output_dir=str(tmp_path),
            heldout=str(heldout),
            n=1,
        )

    # Swallowed: we still got a result despite the persistence blowing up.
    assert result.held_out_loss == pytest.approx(0.25)
    history.record_run_completed.assert_called_once()


# =============================================================================
# evaluate_run — error codes
# =============================================================================

def test_evaluate_run_unknown_run_id_raises_not_found(tmp_path):
    """Unknown run_id -> UserInputError(code=INPUT_EVAL_RUN_NOT_FOUND)."""
    history = MagicMock()
    history.get_run.return_value = None

    with patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        with pytest.raises(UserInputError) as exc_info:
            evaluate_run("nope", output_dir=str(tmp_path))

    assert exc_info.value.code == "INPUT_EVAL_RUN_NOT_FOUND"


def test_evaluate_run_heldout_unresolved_raises(tmp_path):
    """No --heldout and a non-path dataset_info -> INPUT_EVAL_HELDOUT_UNRESOLVED."""
    history = MagicMock()
    # dataset_info is a class name (in-memory dataset) — not a re-splittable path.
    history.get_run.return_value = _run_entry(dataset_info="InMemoryDataset")

    with patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        with pytest.raises(UserInputError) as exc_info:
            evaluate_run(
                "abc123def456",
                output_dir=str(tmp_path),
                heldout=None,
            )

    assert exc_info.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"


def test_evaluate_run_model_load_crash_raises_runtime(tmp_path):
    """A model-load crash surfaces as TrainingError(code=RUNTIME_EVAL_FAILED)."""
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        '{"messages": [{"role": "user", "content": "q"}, '
        '{"role": "assistant", "content": "a"}]}\n',
        encoding="utf-8",
    )
    history = MagicMock()
    history.get_run.return_value = _run_entry()

    def _boom(_run):
        raise TrainingError(
            "model load exploded", code="RUNTIME_EVAL_FAILED"
        )

    with patch.object(ev, "_load_model_and_tokenizer", side_effect=_boom), \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        with pytest.raises(TrainingError) as exc_info:
            evaluate_run(
                "abc123def456",
                output_dir=str(tmp_path),
                heldout=str(heldout),
            )

    assert exc_info.value.code == "RUNTIME_EVAL_FAILED"


# =============================================================================
# Held-out resolution precedence (--heldout wins)
# =============================================================================

def test_heldout_path_wins_over_resplit(tmp_path):
    """--heldout is used verbatim; the run's dataset is NOT re-split."""
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        '{"messages": [{"role": "user", "content": "x"}, '
        '{"role": "assistant", "content": "y z"}]}\n',
        encoding="utf-8",
    )
    # dataset_info points at a different real file; if precedence were wrong the
    # re-split path would fire. We assert the heldout path resolver is used.
    other = tmp_path / "trained.jsonl"
    other.write_text(
        '{"messages": [{"role": "user", "content": "a"}, '
        '{"role": "assistant", "content": "b"}]}\n',
        encoding="utf-8",
    )

    model = _make_model(loss_value=0.5)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry(dataset_info=str(other))

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch.object(ev, "_heldout_texts_from_path", wraps=ev._heldout_texts_from_path) as wrap_path, \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        evaluate_run(
            "abc123def456",
            output_dir=str(tmp_path),
            heldout=str(heldout),
            n=1,
        )

    wrap_path.assert_called_once_with(str(heldout))


def test_heldout_resplit_used_when_no_path(tmp_path):
    """With no --heldout, a path-shaped dataset_info IS re-split (with WARN)."""
    trained = tmp_path / "trained.jsonl"
    # Several rows so the 0.9 split leaves a non-empty test slice.
    lines = "".join(
        f'{{"messages": [{{"role": "user", "content": "u{i}"}}, '
        f'{{"role": "assistant", "content": "answer {i} here"}}]}}\n'
        for i in range(20)
    )
    trained.write_text(lines, encoding="utf-8")

    model = _make_model(loss_value=0.5)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry(dataset_info=str(trained))

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        result = evaluate_run(
            "abc123def456",
            output_dir=str(tmp_path),
            heldout=None,
            n=1,
        )

    # Re-split produced a non-empty held-out slice -> a finite loss.
    assert result.held_out_loss is not None


# =============================================================================
# Held-out resolution via in-memory texts (TRAINER-A-002 eval-gate seam)
# =============================================================================

def test_resolve_heldout_texts_in_memory_wins(tmp_path):
    """In-memory ``heldout_texts`` take precedence over a path + dataset_info,
    and blank entries are filtered out."""
    run = _run_entry(dataset_info=str(tmp_path / "nope.jsonl"))
    texts = ev._resolve_heldout_texts(
        run, heldout="/some/path.jsonl", seed=0,
        heldout_texts=["  hello world  ", "", "   ", "second text"],
    )
    assert texts == ["hello world", "second text"]


def test_resolve_heldout_texts_in_memory_all_blank_raises(tmp_path):
    """An all-empty in-memory held-out list raises the stable unresolved code,
    not a silent empty pass-through."""
    run = _run_entry()
    with pytest.raises(UserInputError) as exc:
        ev._resolve_heldout_texts(run, heldout=None, seed=0, heldout_texts=["", "  "])
    assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"


def test_evaluate_run_uses_in_memory_heldout_texts(tmp_path):
    """evaluate_run with heldout_texts=[...] computes loss against them WITHOUT
    touching the path or dataset re-split resolvers."""
    model = _make_model(loss_value=0.75)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry()

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch.object(ev, "_heldout_texts_from_path") as path_resolver, \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        result = evaluate_run(
            "abc123def456",
            output_dir=str(tmp_path),
            heldout_texts=["a held out text", "another held out text"],
            n=1,
        )

    # The path resolver was never consulted (in-memory texts win).
    path_resolver.assert_not_called()
    # Mean CE of constant 0.75 over the 2 in-memory texts == 0.75.
    assert result.held_out_loss == pytest.approx(0.75)


# =============================================================================
# Device-move atomicity (TRAINER-DATA-002): input_ids + attention_mask must
# never end up split across devices on an asymmetric .to(device) failure.
# =============================================================================

class _DeviceTrackingTensor:
    """A fake tensor tracking which device it currently lives on. ``.to`` can be
    configured to raise (simulating an asymmetric device-move failure)."""

    def __init__(self, rows, device="cpu", raise_on_to=False):
        self._rows = rows
        self.device = device
        self._raise_on_to = raise_on_to

    @property
    def shape(self):
        n_rows = len(self._rows)
        n_cols = len(self._rows[0]) if n_rows else 0
        return (n_rows, n_cols)

    def to(self, device):
        if self._raise_on_to:
            raise RuntimeError("simulated asymmetric device-move failure")
        return _DeviceTrackingTensor(self._rows, device=device)

    def __getitem__(self, idx):
        return _FakeRow(self._rows[idx])


def test_compute_loss_device_move_stays_atomic_on_asymmetric_failure():
    """When attention_mask.to(device) raises, input_ids must NOT have been moved
    either — both reach the model on the SAME (original) device.

    Pre-fix the two moves were committed independently inside one try/except, so
    a failure AFTER input_ids moved but BEFORE attention_mask moved left them on
    different devices. The fix stages both moves and only commits them together.
    """
    captured = {}

    def _forward(**kwargs):
        captured["input_ids"] = kwargs["input_ids"]
        captured["attention_mask"] = kwargs["attention_mask"]
        return SimpleNamespace(loss=_FakeLoss(0.5))

    model = MagicMock()
    model.device = "cuda:0"
    model.eval = MagicMock()
    model.side_effect = _forward

    tok = MagicMock()

    def _tokenize(text, **kwargs):
        return {
            # input_ids moves fine; attention_mask raises on .to(device).
            "input_ids": _DeviceTrackingTensor([[0, 1, 2]], device="cpu"),
            "attention_mask": _DeviceTrackingTensor(
                [[1, 1, 1]], device="cpu", raise_on_to=True
            ),
        }

    tok.side_effect = _tokenize

    with patch.dict(sys.modules, {"torch": _FakeTorch()}):
        loss = ev._compute_held_out_loss(
            model, tok, ["one text"], max_length=64
        )

    assert loss == pytest.approx(0.5)
    # The load-bearing assertion: both tensors reached the forward pass on the
    # SAME device. The asymmetric failure rolled BOTH back to the original "cpu".
    assert captured["input_ids"].device == captured["attention_mask"].device
    assert captured["input_ids"].device == "cpu"


# =============================================================================
# Held-out resolution on a PREFERENCE/ORPO dataset (Phase 8 fix)
# =============================================================================
# Before the FormatConverter PREFERENCE branch landed, a held-out file in
# preference format ({prompt, chosen, rejected}) flattened to 0 texts via
# to_chatml() and evaluate_run raised INPUT_EVAL_HELDOUT_UNRESOLVED blaming the
# wrong cause ("not a readable on-disk file"). The converter fix renders chosen,
# so held-out loss is now computable against prompt->chosen.

def test_heldout_texts_from_preference_file_resolves(tmp_path):
    """_heldout_texts_from_path on a preference jsonl yields real ChatML texts
    (the chosen response), not zero — the eval half of the converter fix."""
    heldout = tmp_path / "pref_heldout.jsonl"
    heldout.write_text(
        '{"prompt": "What is 2+2?", "chosen": "It is four.", "rejected": "It is five."}\n'
        '{"prompt": "Capital of France?", "chosen": "Paris.", "rejected": "London."}\n',
        encoding="utf-8",
    )
    texts = ev._heldout_texts_from_path(str(heldout))
    assert len(texts) == 2
    # chosen is rendered; rejected is NOT part of the held-out text.
    joined = "\n".join(texts)
    assert "It is four." in joined
    assert "Paris." in joined
    assert "It is five." not in joined
    assert "London." not in joined


def test_evaluate_run_on_preference_heldout_computes_loss(tmp_path):
    """End-to-end: evaluate_run with a preference-format --heldout computes a
    held-out loss (no INPUT_EVAL_HELDOUT_UNRESOLVED). Regression for the MEDIUM
    eval finding — preference held-out files used to flatten to 0 texts."""
    heldout = tmp_path / "pref_heldout.jsonl"
    heldout.write_text(
        '{"prompt": "q1", "chosen": "answer one here", "rejected": "bad one"}\n'
        '{"prompt": "q2", "chosen": "answer two here", "rejected": "bad two"}\n',
        encoding="utf-8",
    )

    model = _make_model(loss_value=0.5)
    tokenizer = _make_tokenizer()
    history = MagicMock()
    history.get_run.return_value = _run_entry()
    history.record_run_completed.return_value = {}

    with patch.dict(sys.modules, {"torch": _FakeTorch()}), \
         patch.object(ev, "_load_model_and_tokenizer", return_value=(model, tokenizer)), \
         patch("backpropagate.checkpoints.RunHistoryManager", return_value=history):
        result = evaluate_run(
            "abc123def456",
            output_dir=str(tmp_path),
            heldout=str(heldout),
            n=1,
            seed=0,
        )

    # Loss computed over the 2 preference rows (constant 0.5 mock) -> 0.5,
    # NOT None and NOT a raised UserInputError.
    assert result.held_out_loss == pytest.approx(0.5)


# =============================================================================
# Prompt resolution
# =============================================================================

def test_load_prompts_builtin_when_none():
    prompts = ev._load_prompts(None, n=5)
    assert prompts == DEFAULT_PROMPTS[:5]


def test_load_prompts_truncates_to_n():
    prompts = ev._load_prompts(None, n=2)
    assert len(prompts) == 2


def test_load_prompts_reads_plain_lines(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text("first prompt\nsecond prompt\n\nthird prompt\n", encoding="utf-8")
    prompts = ev._load_prompts(str(p), n=10)
    assert prompts == ["first prompt", "second prompt", "third prompt"]


def test_load_prompts_reads_jsonl_objects(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"prompt": "json prompt one"}\n{"prompt": "json prompt two"}\n',
        encoding="utf-8",
    )
    prompts = ev._load_prompts(str(p), n=10)
    assert prompts == ["json prompt one", "json prompt two"]


def test_load_prompts_missing_file_raises():
    with pytest.raises(UserInputError):
        ev._load_prompts("/nonexistent/prompts.txt", n=5)


# =============================================================================
# eval_gate — accept / reject / equal / names regression
# =============================================================================

def _result(loss):
    return EvalResult(
        run_id="r",
        model_name="m",
        held_out_loss=loss,
        perplexity=None,
        generations=[],
        n_prompts=0,
    )


def test_eval_gate_rejects_when_after_worse():
    """after > before + max_regression -> reject, and names the regression."""
    before = _result(1.0)
    after = _result(1.5)
    decision = eval_gate(before, after, max_regression=0.0)
    assert isinstance(decision, EvalGateDecision)
    assert decision.accept is False
    assert decision.regression == pytest.approx(0.5)
    assert "0.5" in decision.reason  # the regression is named in the reason


def test_eval_gate_accepts_when_equal():
    """Equal loss (regression == 0) accepts at the default max_regression=0."""
    before = _result(1.0)
    after = _result(1.0)
    decision = eval_gate(before, after)
    assert decision.accept is True
    assert decision.regression == pytest.approx(0.0)


def test_eval_gate_accepts_when_improved():
    before = _result(1.0)
    after = _result(0.6)
    decision = eval_gate(before, after)
    assert decision.accept is True
    assert decision.regression == pytest.approx(-0.4)
    assert "improved" in decision.reason.lower()


def test_eval_gate_tolerates_regression_within_budget():
    """A regression within max_regression is accepted."""
    before = _result(1.0)
    after = _result(1.2)
    decision = eval_gate(before, after, max_regression=0.5)
    assert decision.accept is True
    assert decision.regression == pytest.approx(0.2)


def test_eval_gate_missing_loss_fails_safe():
    """Missing held-out loss on either side -> reject (fail-safe)."""
    decision = eval_gate(_result(None), _result(1.0))
    assert decision.accept is False
    # regression is NaN (indeterminate)
    assert decision.regression != decision.regression  # NaN != NaN


# =============================================================================
# diff_evals — row shape
# =============================================================================

def test_diff_evals_row_shape():
    a = EvalResult(
        run_id="run-a",
        model_name="model-a",
        held_out_loss=1.0,
        perplexity=2.718,
        generations=[],
        n_prompts=5,
    )
    b = EvalResult(
        run_id="run-b",
        model_name="model-b",
        held_out_loss=0.8,
        perplexity=2.225,
        generations=[],
        n_prompts=5,
    )
    diff = diff_evals(a, b)
    assert isinstance(diff, EvalDiff)
    assert diff.run_id_a == "run-a"
    assert diff.run_id_b == "run-b"
    # Every row is a 3-tuple (metric_name, value_a, value_b).
    assert all(isinstance(row, tuple) and len(row) == 3 for row in diff.rows)
    metric_names = {row[0] for row in diff.rows}
    assert {"model_name", "held_out_loss", "perplexity", "n_prompts"} <= metric_names
    # Values are pre-formatted strings ready for a table render.
    loss_row = next(r for r in diff.rows if r[0] == "held_out_loss")
    assert loss_row[1] == "1.0000"
    assert loss_row[2] == "0.8000"


def test_diff_evals_handles_none_metrics():
    """None metrics render as 'n/a' rather than crashing the table."""
    a = EvalResult("a", "m", None, None, [], 0)
    b = EvalResult("b", "m", 0.5, 1.6, [], 1)
    diff = diff_evals(a, b)
    loss_row = next(r for r in diff.rows if r[0] == "held_out_loss")
    assert loss_row[1] == "n/a"
    assert loss_row[2] == "0.5000"


def test_eval_diff_to_dict_roundtrips_rows_as_lists():
    a = EvalResult("a", "m", 1.0, 2.7, [], 3)
    b = EvalResult("b", "m", 0.9, 2.4, [], 3)
    d = diff_evals(a, b).to_dict()
    assert d["run_id_a"] == "a"
    assert d["run_id_b"] == "b"
    # tuples serialized to lists so the dict is JSON-clean.
    assert all(isinstance(row, list) for row in d["rows"])


# =============================================================================
# EvalResult.to_dict shape
# =============================================================================

def test_eval_result_to_dict_shape():
    result = EvalResult(
        run_id="rid",
        model_name="mn",
        held_out_loss=0.5,
        perplexity=1.6,
        generations=[GenerationSample(prompt="p", completion="c")],
        n_prompts=1,
    )
    d = result.to_dict()
    assert d["run_id"] == "rid"
    assert d["model_name"] == "mn"
    assert d["held_out_loss"] == 0.5
    assert d["perplexity"] == 1.6
    assert d["n_prompts"] == 1
    assert d["generations"] == [{"prompt": "p", "completion": "c"}]


# =============================================================================
# _load_model_and_tokenizer — real load path with sys.modules fakes
# =============================================================================

def test_load_model_and_tokenizer_wraps_failures_as_runtime_eval_failed():
    """A transformers import/load failure is wrapped as RUNTIME_EVAL_FAILED."""
    # Inject a fake transformers whose from_pretrained raises so we exercise the
    # except branch deterministically without a real download.
    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained.side_effect = OSError(
        "no network"
    )

    with patch.dict(
        sys.modules,
        {"torch": _FakeTorch(), "transformers": fake_transformers},
    ):
        with pytest.raises(TrainingError) as exc_info:
            ev._load_model_and_tokenizer(_run_entry())

    assert exc_info.value.code == "RUNTIME_EVAL_FAILED"
    # The original cause is chained for traceback fidelity.
    assert isinstance(exc_info.value.__cause__, OSError)


def test_load_model_and_tokenizer_base_only_when_no_checkpoint():
    """With no checkpoint dir, the bare base model is returned (no peft attach)."""
    fake_transformers = MagicMock()
    fake_model = MagicMock()
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "</s>"
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = fake_model
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok

    # peft must NOT be needed when there is no checkpoint path; inject a fake
    # that would explode if PeftModel.from_pretrained were called.
    fake_peft = MagicMock()
    fake_peft.PeftModel.from_pretrained.side_effect = AssertionError(
        "peft should not be touched when there is no adapter dir"
    )

    with patch.dict(
        sys.modules,
        {
            "torch": _FakeTorch(),
            "transformers": fake_transformers,
            "peft": fake_peft,
        },
    ):
        model, tokenizer = ev._load_model_and_tokenizer(
            _run_entry(checkpoint_path=None)
        )

    assert model is fake_model
    assert tokenizer is fake_tok
    # pad_token fell back to eos_token when missing.
    assert tokenizer.pad_token == "</s>"
    fake_peft.PeftModel.from_pretrained.assert_not_called()
