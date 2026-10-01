"""Tests for how ``MultiRunTrainer`` reads/writes adapter weights and scores the model:
``_prepare_for_next_run``, ``_load_lora_state_dict``, ``_verify_peft_api``,
``_load_resume_checkpoint`` and ``_compute_validation_loss``.

Real: a tiny PEFT/LoRA adapter over a tiny Llama, the tiny tokenizer, real
adapter files written by ``save_pretrained`` / ``torch.save`` /
``safetensors``. Where a stand-in object replaces the model (PEFT-API
presence checks, corrupt-model defensive paths) the docstring says so.
"""

from __future__ import annotations

import logging
import math
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate.exceptions import BackpropagateError
from backpropagate.multi_run import MergeMode, MultiRunTrainer
from backpropagate.slao import SLAOConfig, SLAOMerger
from tests.helpers.tiny_models import tiny_llama, tiny_tokenizer
from tests.test_multi_run_cov_support import (
    FakeInnerTrainer,
    build_peft_llama,
    fill_adapter,
    lora_params,
)

MR_LOGGER = "backpropagate.multi_run"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _mrt(model, **cfg):
    from backpropagate.multi_run import MultiRunConfig

    mrt = MultiRunTrainer(model="m", config=MultiRunConfig(**cfg))
    mrt._trainer = FakeInnerTrainer(model)
    return mrt


# =============================================================================
# _prepare_for_next_run
# =============================================================================


class TestPrepareForNextRun:
    def test_unseeded_merger_leaves_the_model_untouched(self):
        model = build_peft_llama()
        mrt = _mrt(model)
        mrt._slao_merger = SLAOMerger(SLAOConfig())  # never initialised: no init weights
        before = lora_params(model)

        mrt._prepare_for_next_run(2)

        for k, v in lora_params(model).items():
            assert torch.equal(v, before[k])

    def test_seeded_merger_reinitialises_a_orthogonally_and_copies_b(self):
        model = build_peft_llama()
        mrt = _mrt(model)
        mrt._slao_merger = SLAOMerger(SLAOConfig())
        fill_adapter(model, run=3, b_value=0.75)
        mrt._slao_merger.initialize(lora_params(model))
        fill_adapter(model, run=9, b_value=-4.0)  # a later, different adapter in the model

        mrt._prepare_for_next_run(2)

        for name, p in model.named_parameters():
            if ".lora_B." in name:
                assert torch.all(p == 0.75), name
            elif ".lora_A." in name:
                assert torch.allclose(p @ p.T, torch.eye(p.shape[0]), atol=1e-5), name

    def test_simple_mode_keeps_the_current_weights(self):
        model = build_peft_llama()
        mrt = _mrt(model, merge_mode=MergeMode.SIMPLE)
        before = lora_params(model)

        mrt._prepare_for_next_run(2)

        for k, v in lora_params(model).items():
            assert torch.equal(v, before[k])


# =============================================================================
# _load_lora_state_dict
# =============================================================================


class TestLoadLoraStateDict:
    def test_manual_load_copies_values_into_the_live_adapter(self):
        model = build_peft_llama()
        mrt = _mrt(model)
        target = {k: torch.full_like(v, 0.25) for k, v in lora_params(model).items()}

        mrt._load_lora_state_dict(target)

        for k, v in lora_params(model).items():
            assert torch.all(v == 0.25), k

    def test_partial_key_overlap_loads_what_matches_and_ignores_the_rest(self):
        model = build_peft_llama()
        mrt = _mrt(model)
        params = lora_params(model)
        b_key = next(k for k in params if ".lora_B." in k)
        before = lora_params(model)

        mrt._load_lora_state_dict({b_key: torch.full_like(params[b_key], 2.0), "ghost.lora_A.w": torch.ones(1)})

        after = lora_params(model)
        assert torch.all(after[b_key] == 2.0)
        for k in params:
            if k != b_key:
                assert torch.equal(after[k], before[k])

    def test_total_key_mismatch_fails_loudly(self):
        mrt = _mrt(build_peft_llama())
        with pytest.raises(BackpropagateError) as exc_info:
            mrt._load_lora_state_dict({"does.not.exist.lora_A.weight": torch.ones(2)})

        err = exc_info.value
        assert err.code == "PEFT_API_INCOMPATIBLE"
        assert err.details["accumulator_keys"] == 1 and err.details["matched_keys"] == 0
        assert err.retryable is False

    def test_empty_state_dict_is_a_noop(self):
        mrt = _mrt(build_peft_llama())
        mrt._load_lora_state_dict({})  # nothing to apply, nothing to complain about

    def test_native_peft_loader_is_preferred_when_present(self):
        """Stand-in model: exposes the PEFT ``load_adapter_state_dict`` API."""

        class _Model:
            received = None

            def load_adapter_state_dict(self, sd):
                _Model.received = sd

        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=_Model())
        state = {"x.lora_A.w": torch.ones(1)}

        mrt._load_lora_state_dict(state)

        assert _Model.received is state


# =============================================================================
# _verify_peft_api
# =============================================================================


class TestVerifyPeftApi:
    def test_extraction_exception_is_wrapped(self):
        class _Model:
            def get_adapter_state_dict(self):
                raise RuntimeError("peft internals changed")

        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=_Model())

        with pytest.raises(BackpropagateError) as exc_info:
            mrt._verify_peft_api()

        err = exc_info.value
        assert err.code == "PEFT_API_INCOMPATIBLE"
        assert "peft internals changed" in err.details["reason"]
        assert isinstance(err.__cause__, RuntimeError)

    def test_model_without_any_lora_parameters_is_rejected(self):
        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=torch.nn.Linear(2, 2))  # plain, no LoRA

        with pytest.raises(BackpropagateError) as exc_info:
            mrt._verify_peft_api()

        assert exc_info.value.code == "PEFT_API_INCOMPATIBLE"
        assert exc_info.value.details == {"extracted_param_count": 0}

    def test_a_only_adapter_passes_with_a_warning(self, caplog):
        model = torch.nn.Module()
        model.layer = torch.nn.Module()
        # parameter ``layer.lora_A.default.weight``: an A matrix with no B partner
        model.layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})
        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=model)

        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            mrt._verify_peft_api()

        assert any("A_keys=1 and B_keys=0" in r.getMessage() for r in caplog.records)

    def test_real_peft_adapter_passes_silently(self, caplog):
        mrt = _mrt(build_peft_llama())
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            mrt._verify_peft_api()
        assert not caplog.records


# =============================================================================
# _load_resume_checkpoint
# =============================================================================


class TestLoadResumeCheckpoint:
    def test_missing_directory(self, tmp_path):
        mrt = _mrt(build_peft_llama())
        with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
            mrt._load_resume_checkpoint(str(tmp_path / "nope"))

    def test_real_peft_load_adapter_restores_the_saved_weights(self, tmp_path):
        saved = build_peft_llama()
        fill_adapter(saved, run=5)
        saved.save_pretrained(str(tmp_path))

        fresh = build_peft_llama()  # different (zero-B) adapter
        mrt = _mrt(fresh)
        mrt._load_resume_checkpoint(str(tmp_path))

        expected = lora_params(saved)
        for k, v in lora_params(fresh).items():
            assert torch.equal(v, expected[k]), k

    def test_load_adapter_is_called_with_the_default_trainable_adapter(self, tmp_path):
        """Stand-in model: records the PEFT ``load_adapter`` call."""
        calls = []

        class _Model:
            def load_adapter(self, path, adapter_name, is_trainable):
                calls.append((path, adapter_name, is_trainable))

        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=_Model())

        mrt._load_resume_checkpoint(str(tmp_path))

        assert calls == [(str(tmp_path), "default", True)]

    def test_fallback_reads_a_torch_bin_state_dict(self, tmp_path):
        """``load_adapter`` unavailable -> the ``.bin`` weights are applied by name."""
        model = build_peft_llama()
        target = {k: torch.full_like(v, 1.5) for k, v in lora_params(model).items()}
        torch.save(target, tmp_path / "adapter_model.bin")
        mrt = MultiRunTrainer(model="m")
        # hide the PEFT loader so the fallback path is the one exercised
        mrt._trainer = FakeInnerTrainer(_NoLoadAdapter(model))

        mrt._load_resume_checkpoint(str(tmp_path))

        for k, v in lora_params(model).items():
            assert torch.all(v == 1.5), k

    def test_fallback_after_a_failing_load_adapter(self, tmp_path, caplog):
        """``load_adapter`` raises -> falls back to the safetensors file."""
        from safetensors.torch import save_file

        model = build_peft_llama()
        target = {k: torch.full_like(v, -0.5) for k, v in lora_params(model).items()}
        save_file(target, str(tmp_path / "adapter_model.safetensors"))
        mrt = MultiRunTrainer(model="m")
        mrt._trainer = FakeInnerTrainer(_FailingLoadAdapter(model))

        with caplog.at_level(logging.DEBUG, logger=MR_LOGGER):
            mrt._load_resume_checkpoint(str(tmp_path))

        assert any("load_adapter path failed" in r.getMessage() for r in caplog.records)
        for k, v in lora_params(model).items():
            assert torch.all(v == -0.5), k

    def test_no_adapter_weights_in_the_directory(self, tmp_path):
        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=_NoLoadAdapter(build_peft_llama()))
        with pytest.raises(FileNotFoundError, match="No adapter weights"):
            mrt._load_resume_checkpoint(str(tmp_path))

    def test_safetensors_file_needs_the_safetensors_package(self, tmp_path, monkeypatch):
        (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
        monkeypatch.setitem(sys.modules, "safetensors.torch", None)  # import -> ImportError
        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=_NoLoadAdapter(build_peft_llama()))

        with pytest.raises(ImportError, match="safetensors is required to load"):
            mrt._load_resume_checkpoint(str(tmp_path))

    def test_bin_checkpoint_does_not_need_safetensors(self, tmp_path, monkeypatch):
        model = build_peft_llama()
        torch.save({k: torch.zeros_like(v) for k, v in lora_params(model).items()},
                   tmp_path / "adapter_model.bin")
        monkeypatch.setitem(sys.modules, "safetensors.torch", None)
        mrt = MultiRunTrainer(model="m")
        mrt._trainer = types.SimpleNamespace(_model=_NoLoadAdapter(model))
        fill_adapter(model, run=2)

        mrt._load_resume_checkpoint(str(tmp_path))

        for k, v in lora_params(model).items():
            assert torch.all(v == 0), k


class _NoLoadAdapter:
    """Proxy hiding ``load_adapter`` so the fallback path is taken (PEFT API boundary)."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        if name == "load_adapter":
            raise AttributeError(name)
        return getattr(self._model, name)


class _FailingLoadAdapter(_NoLoadAdapter):
    def __getattr__(self, name):
        if name == "load_adapter":
            def boom(*_a, **_k):
                raise ValueError("adapter incompatible")

            return boom
        return getattr(self._model, name)


# =============================================================================
# _compute_validation_loss (real forward passes on the tiny Llama)
# =============================================================================


def _val_mrt(model, tokenizer, *, validation_samples=3, **cfg):
    from backpropagate.multi_run import MultiRunConfig

    mrt = MultiRunTrainer(
        model="m",
        config=MultiRunConfig(
            validate_every_run=True, validation_samples=validation_samples, **cfg
        ),
    )
    mrt._trainer = types.SimpleNamespace(_model=model, _tokenizer=tokenizer, max_seq_length=32)
    return mrt


def _expected_loss(model, tokenizer, texts):
    losses = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
            losses.append(model(**enc, labels=enc["input_ids"]).loss.item())
    model.train()
    return sum(losses) / len(losses)


class TestComputeValidationLoss:
    def test_text_rows_average_the_per_sample_loss_over_the_holdout(self):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        texts = [f"the cat sat on the mat {w}" for w in ("dog", "park", "yes", "no", "two")]
        ds = Dataset.from_dict({"text": texts})
        mrt = _val_mrt(model, tok, validation_samples=2)

        loss = mrt._compute_validation_loss(ds, 1)

        # 5 rows -> holdout = max(int(5 * 0.1), 1) = 1 row (the last)
        assert loss == pytest.approx(_expected_loss(model, tok, texts[-1:]), abs=1e-5)
        assert model.training is True  # returned to training mode

    def test_holdout_is_capped_by_validation_samples(self):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        texts = [f"a cat ran to the park {i % 3}" for i in range(40)]
        texts = [t.replace("0", "yes").replace("1", "no").replace("2", "two") for t in texts]
        ds = Dataset.from_dict({"text": texts})
        mrt = _val_mrt(model, tok, validation_samples=2)

        loss = mrt._compute_validation_loss(ds, 1)

        # holdout = last 4 rows, only the first validation_samples=2 of them are used
        assert loss == pytest.approx(_expected_loss(model, tok, texts[36:38]), abs=1e-5)

    def test_chat_message_rows_go_through_the_chat_template(self):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        convo = [{"role": "user", "content": "what is two plus four"},
                 {"role": "assistant", "content": "yes"}]
        ds = Dataset.from_dict({"messages": [convo] * 10})
        mrt = _val_mrt(model, tok, validation_samples=1)

        loss = mrt._compute_validation_loss(ds, 1)

        rendered = tok.apply_chat_template(convo, tokenize=False)
        assert loss == pytest.approx(_expected_loss(model, tok, [rendered]), abs=1e-5)

    def test_sharegpt_rows_are_joined_with_newlines(self):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        convo = [{"from": "human", "value": "what is two"}, {"from": "gpt", "value": "yes"}]
        ds = Dataset.from_dict({"conversations": [convo] * 10})
        mrt = _val_mrt(model, tok, validation_samples=1)

        loss = mrt._compute_validation_loss(ds, 1)

        assert loss == pytest.approx(_expected_loss(model, tok, ["what is two\nyes"]), abs=1e-5)

    def test_rows_without_any_text_field_are_skipped_and_counted(self, caplog):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        ds = Dataset.from_dict({"unrelated": list(range(20))})
        mrt = _val_mrt(model, tok, validation_samples=5)

        with caplog.at_level(logging.INFO, logger=MR_LOGGER):
            loss = mrt._compute_validation_loss(ds, 4)

        assert loss == float("inf")  # nothing could be evaluated
        msgs = [r.getMessage() for r in caplog.records]
        assert any("silently skipped 2 samples (100.0% of 2)" in m and "(run 4)" in m
                   for m in msgs)
        assert any("No validation samples were successfully evaluated" in m for m in msgs)
        # a material (>10%) silent skip is escalated to WARNING
        assert any(r.levelno == logging.WARNING and "silently skipped" in r.getMessage()
                   for r in caplog.records)

    def test_small_silent_skip_is_only_informational(self, caplog):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        # 100 rows, holdout 10; only the first holdout row lacks text -> 10%: not > 10%
        rows = [{"text": "the cat sat", "extra": 0} for _ in range(100)]
        ds = Dataset.from_list(rows)
        mrt = _val_mrt(model, tok, validation_samples=50)

        class _FlakySelect:
            """Real rows, except the first holdout row hides its text column."""

            def __init__(self, base):
                self._base = base

            def __len__(self):
                return len(self._base)

            def select(self, idx):
                picked = [dict(r) for r in self._base.select(idx)]
                del picked[0]["text"]
                picked[0]["other"] = 1
                return picked

        with caplog.at_level(logging.INFO, logger=MR_LOGGER):
            loss = mrt._compute_validation_loss(_FlakySelect(ds), 1)

        assert math.isfinite(loss)
        skip_records = [r for r in caplog.records if "silently skipped" in r.getMessage()]
        assert len(skip_records) == 1
        assert skip_records[0].levelno == logging.INFO
        assert "1 samples (10.0% of 10)" in skip_records[0].getMessage()

    def test_unreadable_rows_are_skipped_with_a_warning_but_good_rows_still_count(
        self, caplog
    ):
        from datasets import Dataset

        model, tok = tiny_llama(layers=1), tiny_tokenizer()
        good = "the cat sat on the mat"
        # holdout = last 2 of 20 rows: one unreadable (None text), one fine
        texts = [good] * 18 + [None, good]
        ds = Dataset.from_dict({"text": texts})
        mrt = _val_mrt(model, tok, validation_samples=5)

        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            loss = mrt._compute_validation_loss(ds, 1)

        assert loss == pytest.approx(_expected_loss(model, tok, [good]), abs=1e-5)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("Skipped validation sample" in m for m in msgs)
        assert any("Skipped 1 validation samples due to errors" in m for m in msgs)

    def test_missing_model_yields_infinite_loss(self, caplog):
        mrt = _val_mrt(None, tiny_tokenizer())
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            assert mrt._compute_validation_loss(_one_row_dataset(), 2) == float("inf")
        assert any("model is None for run 2" in r.getMessage() for r in caplog.records)

    def test_corrupt_model_is_survivable(self, caplog):
        """Stand-in: an object lacking ``eval`` / ``train`` (a half-built model)."""
        mrt = _val_mrt(types.SimpleNamespace(), tiny_tokenizer())
        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            loss = mrt._compute_validation_loss(_one_row_dataset(), 1)

        assert loss == float("inf")
        msgs = [r.getMessage() for r in caplog.records]
        assert any("model.eval() raised AttributeError" in m for m in msgs)
        assert any("model.train() restore raised AttributeError" in m for m in msgs)


def _one_row_dataset():
    from datasets import Dataset

    return Dataset.from_dict({"text": ["the cat sat"] * 10})
