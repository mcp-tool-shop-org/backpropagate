"""Regression tests for bugs found while raising ``trainer.py`` coverage.

``Trainer.load_model`` caught every non-trust-remote-code ``ModelLoadError``
raised by the Unsloth loader in its catch-all and wrapped it again, so the
F-019 cause category (auth / not_found / network / version) the loader had
just computed was reset to ``"unknown"`` and the message prefix doubled.

Mock boundary: ``unsloth`` (CUDA-only optional library; a fake module whose
``from_pretrained`` / ``get_peft_model`` fail on demand) and the network
(the failure itself is injected; retry sleeps are zeroed).
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from backpropagate import trainer as T  # noqa: E402
from backpropagate.exceptions import ModelLoadError  # noqa: E402
from backpropagate.trainer import Trainer  # noqa: E402
from tests.helpers.tiny_models import sentences, tiny_llama, tiny_tokenizer  # noqa: E402


class _FakeFastLanguageModel:
    from_pretrained_exc: BaseException | None = None
    peft_exc: BaseException | None = None

    @classmethod
    def from_pretrained(cls, **kw):
        if cls.from_pretrained_exc is not None:
            raise cls.from_pretrained_exc
        return tiny_llama(layers=1), tiny_tokenizer()

    @classmethod
    def get_peft_model(cls, model, **kw):
        if cls.peft_exc is not None:
            raise cls.peft_exc
        return model


@pytest.fixture
def unsloth_without_fallback(monkeypatch):
    _FakeFastLanguageModel.from_pretrained_exc = None
    _FakeFastLanguageModel.peft_exc = None
    mod = types.ModuleType("unsloth")
    mod.FastLanguageModel = _FakeFastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", mod)
    monkeypatch.setattr(T, "check_feature", lambda name: name == "unsloth")
    monkeypatch.setattr(T, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(T, "_RETRY_MAX_SECONDS", 0)
    yield _FakeFastLanguageModel
    _FakeFastLanguageModel.from_pretrained_exc = None
    _FakeFastLanguageModel.peft_exc = None


def _trainer():
    return Trainer(model="acme/Tiny-1B", use_unsloth=True, unsloth_fallback=False,
                   batch_size=2, report_to="none")


def test_unsloth_network_failure_keeps_its_network_category(unsloth_without_fallback):
    unsloth_without_fallback.from_pretrained_exc = ConnectionError("hub unreachable")
    with pytest.raises(ModelLoadError) as ei:
        _trainer().load_model()
    assert ei.value.cause_category == "network"
    assert ei.value.details["cause_category"] == "network"
    # raised once, not wrapped a second time
    assert str(ei.value).count("Failed to load model") == 1
    assert "Unsloth model loading failed: hub unreachable" in ei.value.reason
    assert isinstance(ei.value.__cause__, ConnectionError)


def test_unsloth_hub_auth_failure_keeps_its_auth_category(unsloth_without_fallback):
    import httpx
    from huggingface_hub.utils import HfHubHTTPError

    req = httpx.Request("GET", "https://huggingface.co/x")
    unsloth_without_fallback.from_pretrained_exc = HfHubHTTPError(
        "gated repo", response=httpx.Response(403, request=req))
    with pytest.raises(ModelLoadError) as ei:
        _trainer().load_model()
    assert ei.value.cause_category == "auth"


def test_peft_application_failure_keeps_its_version_category(unsloth_without_fallback):
    unsloth_without_fallback.peft_exc = ImportError("peft too old")
    t = _trainer()
    with pytest.raises(ModelLoadError) as ei:
        t.load_model()
    assert ei.value.cause_category == "version"
    assert "Failed to apply LoRA: peft too old" in ei.value.reason
    assert t._is_loaded is False


def test_non_modelload_unsloth_failures_are_still_wrapped_and_classified(unsloth_without_fallback):
    """The catch-all keeps wrapping everything that is NOT already a ModelLoadError."""
    with patch.object(Trainer, "_load_with_unsloth", side_effect=KeyError("boom")):
        with pytest.raises(ModelLoadError) as ei:
            _trainer().load_model()
    assert ei.value.cause_category == "unknown"
    assert isinstance(ei.value.__cause__, KeyError)


# ---------------------------------------------------------------------------
# Single-run resume_from used to hand HF the run's OUTPUT dir
# ---------------------------------------------------------------------------

def test_resolve_resume_checkpoint_prefers_a_checkpoint_itself_then_newest_child(tmp_path):
    root = tmp_path / "out"
    for d in ("checkpoint-2", "checkpoint-10", "checkpoint-9"):
        (root / d).mkdir(parents=True)
        (root / d / "trainer_state.json").write_text("{}")
    assert T._resolve_resume_checkpoint(str(root)).endswith("checkpoint-10")  # numeric, not lexical
    assert T._resolve_resume_checkpoint(str(root / "checkpoint-2")).endswith("checkpoint-2")


def test_resolve_resume_checkpoint_returns_none_when_nothing_to_resume(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert T._resolve_resume_checkpoint(str(empty)) is None
    f = tmp_path / "a_file"
    f.write_text("x")
    assert T._resolve_resume_checkpoint(str(f)) is None


def test_single_run_resume_continues_from_the_latest_checkpoint(tmp_path, monkeypatch):
    """Real CPU LoRA training: train 3 steps, then resume to step 5.

    Before the fix the run history's output-dir path went straight to
    ``Trainer.train(resume_from_checkpoint=<output_dir>)`` and HF raised
    ``FileNotFoundError: <output_dir>/trainer_state.json``, so every
    single-run ``resume_from`` of a finished run failed.
    """
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(T.settings.training, "save_steps", 2)
    monkeypatch.setattr(T.settings.training, "logging_steps", 1)
    model_dir = tmp_path / "m"
    tiny_llama(layers=2).save_pretrained(model_dir)
    tiny_tokenizer().save_pretrained(model_dir)
    data = tmp_path / "d.jsonl"
    with open(data, "w", encoding="utf-8") as fh:
        for s in sentences(16):
            fh.write(json.dumps({"messages": [{"role": "user", "content": s},
                                              {"role": "assistant", "content": s}]}) + "\n")
    kw = {
        "model": str(model_dir), "use_unsloth": False, "load_in_4bit": False, "batch_size": 2,
        "max_seq_length": 32, "output_dir": str(tmp_path / "out"), "learning_rate": 1e-3,
        "packing": False, "report_to": "none", "lora_r": 4,
    }
    first = Trainer(**kw).train(str(data), steps=3)
    assert len(first.loss_history) == 3

    resumed = Trainer(**kw).train(str(data), steps=5, resume_from=first.run_id)
    assert resumed.run_id == first.run_id
    # HF continued from checkpoint-3: its trainer_state carried the first 3 losses
    # and only 2 new steps ran, so the history grows 3 -> 5 with the old prefix intact.
    assert len(resumed.loss_history) == 5
    assert resumed.loss_history[:3] == first.loss_history
    state = json.loads((tmp_path / "out" / "checkpoint-5" / "trainer_state.json").read_text())
    assert state["global_step"] == 5
