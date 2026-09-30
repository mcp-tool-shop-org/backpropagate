"""``trust_remote_code`` is OFF by default and is one setting for every loader.

Loading a Hugging Face model repo with ``trust_remote_code=True`` executes the
Python files that repo ships. The default is therefore False; an operator opts
in with ``BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE=true`` (or
``settings.model.trust_remote_code = True`` in Python). When a repo needs the
code and the setting is off, the load raises a structured
``CONFIG_TRUST_REMOTE_CODE_REQUIRED`` error that names the model and the exact
opt-in instead of a bare transformers ``ValueError``.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import backpropagate.config as real_cfg
from backpropagate.exceptions import (
    ERROR_CODES,
    ModelLoadError,
    TrainingError,
    TrustRemoteCodeRequiredError,
    is_trust_remote_code_error,
)

NEEDS_CODE = ValueError(
    "The repository org/custom-model contains custom code which must be "
    "executed to correctly load the model. You can inspect the repository "
    "content at https://hf.co/org/custom-model .\n"
    "Please pass the argument `trust_remote_code=True` to allow custom code "
    "to be run."
)


@pytest.fixture
def trust_setting(monkeypatch):
    """Set ``settings.model.trust_remote_code`` for one test."""

    def _set(value: bool) -> None:
        # Other tests call ``reload_settings()``, which rebinds
        # ``config.settings`` while ``trainer.settings`` keeps the old object
        # (eval/datasets resolve ``config.settings`` at call time). Set both so
        # the test does not depend on test order.
        import backpropagate.trainer as trainer_mod

        for obj in {id(real_cfg.settings): real_cfg.settings,
                    id(trainer_mod.settings): trainer_mod.settings}.values():
            monkeypatch.setattr(obj.model, "trust_remote_code", value)

    _set(False)
    return _set


def _fake_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    return tok


# ---------------------------------------------------------------------------
# Default is False in both config branches
# ---------------------------------------------------------------------------


class TestDefaultIsOff:
    def test_pydantic_or_active_branch_default(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", raising=False)
        assert real_cfg.ModelConfig().trust_remote_code is False

    def test_dataclass_fallback_default(self, monkeypatch):
        monkeypatch.delenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", raising=False)
        sys.modules.setdefault("backpropagate", importlib.import_module("backpropagate"))
        source = Path(real_cfg.__file__).read_text(encoding="utf-8")
        fake = types.ModuleType("backpropagate._config_trust_probe")
        fake.__dict__["__file__"] = real_cfg.__file__
        fake.__dict__["__name__"] = "backpropagate.config"
        with patch.dict(sys.modules, {"pydantic_settings": None}):
            exec(compile(source, real_cfg.__file__, "exec"), fake.__dict__)  # noqa: S102
        assert fake.__dict__["PYDANTIC_SETTINGS_AVAILABLE"] is False
        assert fake.__dict__["ModelConfig"]().trust_remote_code is False

    def test_env_var_opts_in(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "true")
        assert real_cfg.ModelConfig().trust_remote_code is True


# ---------------------------------------------------------------------------
# The one setting reaches every loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [False, True])
class TestSettingReachesLoaders:
    def test_transformers_4bit_path(self, trust_setting, value):
        from backpropagate.trainer import Trainer

        trust_setting(value)
        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModelForCausalLM.from_pretrained") as m_model, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()) as m_tok, \
             patch("transformers.BitsAndBytesConfig"), \
             patch("peft.prepare_model_for_kbit_training", return_value=MagicMock()), \
             patch("peft.get_peft_model", return_value=MagicMock()), \
             patch("peft.LoraConfig"):
            Trainer(use_unsloth=False)._load_with_transformers()

        assert m_model.call_args.kwargs["trust_remote_code"] is value
        assert m_tok.call_args.kwargs["trust_remote_code"] is value

    def test_transformers_full_finetune_path(self, trust_setting, value):
        from backpropagate.trainer import Trainer

        trust_setting(value)
        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModelForCausalLM.from_pretrained") as m_model, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()) as m_tok:
            trainer = Trainer(use_unsloth=False, mode="full", model="Qwen/Qwen2.5-0.5B-Instruct")
            trainer._load_with_transformers()

        assert m_model.call_args.kwargs["trust_remote_code"] is value
        assert m_tok.call_args.kwargs["trust_remote_code"] is value

    def test_unsloth_path(self, trust_setting, value):
        from backpropagate import feature_flags
        from backpropagate.trainer import Trainer

        trust_setting(value)
        fast_lm = MagicMock()
        # A real module tree, not a MagicMock: since #230 the trainer derives
        # the "all-linear" target list from the loaded model's nn.Linear
        # layers, and raises when it finds none.
        import torch

        class _Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4)

        fast_lm.from_pretrained.return_value = (_Tiny(), MagicMock())
        fast_lm.get_peft_model.return_value = MagicMock()
        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}), \
             patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=fast_lm)}):
            Trainer(use_unsloth=True)._load_with_unsloth()

        assert fast_lm.from_pretrained.call_args.kwargs["trust_remote_code"] is value

    def test_eval_loader(self, trust_setting, value):
        from backpropagate.eval import _load_model_and_tokenizer

        trust_setting(value)
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as m_model, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()) as m_tok:
            _load_model_and_tokenizer({"model_name": "org/m", "run_id": "r1"})

        assert m_model.call_args.kwargs["trust_remote_code"] is value
        assert m_tok.call_args.kwargs["trust_remote_code"] is value

    def test_perplexity_filter_loader(self, trust_setting, value):
        from backpropagate.datasets import PerplexityFilter

        trust_setting(value)
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as m_model, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()) as m_tok:
            PerplexityFilter(model_name="gpt2", device="cpu")._load_model()

        assert m_model.call_args.kwargs["trust_remote_code"] is value
        assert m_tok.call_args.kwargs["trust_remote_code"] is value


class TestMlxRail:
    def _backend(self, **kw):
        from backpropagate.mlx_backend import MLXBackend

        return MLXBackend(
            model="mlx-community/m", dataset_dir="d", adapter_path="a",
            lora_r=16, lora_alpha=32, lora_dropout=0.05, learning_rate=1e-5,
            iters=10, batch_size=1, max_seq_length=128, **kw,
        )

    def test_default_config_is_unchanged(self):
        assert "trust_remote_code" not in self._backend().build_config()

    def test_opt_in_reaches_mlx_lm_config(self):
        assert self._backend(trust_remote_code=True).build_config()["trust_remote_code"] is True

    def test_trainer_threads_setting_to_backend(self):
        # The only construction site of MLXBackend passes the setting.
        src = Path(real_cfg.__file__).with_name("trainer.py").read_text(encoding="utf-8")
        assert "trust_remote_code=settings.model.trust_remote_code," in src.split("MLXBackend(")[1][:600]


# ---------------------------------------------------------------------------
# Structured error when the repo needs code and the setting is off
# ---------------------------------------------------------------------------


class TestStructuredError:
    def test_code_is_in_catalog(self):
        assert "CONFIG_TRUST_REMOTE_CODE_REQUIRED" in ERROR_CODES

    def test_error_content(self):
        err = TrustRemoteCodeRequiredError("org/custom-model")
        assert isinstance(err, ModelLoadError) and isinstance(err, TrainingError)
        assert err.code == "CONFIG_TRUST_REMOTE_CODE_REQUIRED"
        assert err.retryable is False
        assert "org/custom-model" in str(err)
        assert "execute Python code from the model repository" in str(err)
        hint = err.suggestion or ""
        assert "BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE=true" in hint
        assert "settings.model.trust_remote_code = True" in hint
        assert "https://huggingface.co/org/custom-model" in hint

    def test_plain_model_load_error_code_unchanged(self):
        err = ModelLoadError("m", "boom")
        assert err.code == "DEP_MODEL_LOAD_FAILED" and err.retryable is True

    def test_transformers_path_raises_structured_error(self, trust_setting):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=NEEDS_CODE), \
             patch("transformers.BitsAndBytesConfig"):
            trainer = Trainer(model="org/custom-model", use_unsloth=False)
            with pytest.raises(TrustRemoteCodeRequiredError) as exc:
                trainer.load_model()
        assert exc.value.model_name == "org/custom-model"
        assert exc.value.__cause__ is NEEDS_CODE

    def test_unsloth_path_raises_and_does_not_fall_back(self, trust_setting):
        from backpropagate import feature_flags
        from backpropagate.trainer import Trainer

        fast_lm = MagicMock()
        # Unsloth re-wraps transformers' ValueError; the chain walk must see it.
        wrapped = RuntimeError("unsloth load failed")
        wrapped.__cause__ = NEEDS_CODE
        fast_lm.from_pretrained.side_effect = wrapped
        with patch("torch.cuda.is_available", return_value=False), \
             patch.dict(feature_flags.FEATURES, {"unsloth": True}), \
             patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=fast_lm)}):
            trainer = Trainer(model="org/custom-model", use_unsloth=True)
            with patch.object(trainer, "_load_with_transformers") as fallback:
                with pytest.raises(TrustRemoteCodeRequiredError):
                    trainer.load_model()
        fallback.assert_not_called()

    def test_eval_raises_structured_error(self, trust_setting):
        from backpropagate.eval import _load_model_and_tokenizer

        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=NEEDS_CODE):
            with pytest.raises(TrustRemoteCodeRequiredError):
                _load_model_and_tokenizer({"model_name": "org/custom-model", "run_id": "r"})

    def test_unrelated_value_error_is_not_misclassified(self):
        assert not is_trust_remote_code_error(ValueError("bad dtype"))
        assert not is_trust_remote_code_error(RuntimeError("trust_remote_code in a RuntimeError"))

    def test_opt_in_loads_without_error(self, trust_setting):
        from backpropagate.trainer import Trainer

        trust_setting(True)
        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()) as m_model, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=_fake_tokenizer()), \
             patch("transformers.BitsAndBytesConfig"), \
             patch("peft.prepare_model_for_kbit_training", return_value=MagicMock()), \
             patch("peft.get_peft_model", return_value=MagicMock()), \
             patch("peft.LoraConfig"):
            trainer = Trainer(model="org/custom-model", use_unsloth=False)
            trainer.load_model()
        assert trainer._is_loaded
        assert m_model.call_args.kwargs["trust_remote_code"] is True


class TestNeverPrompts:
    """A non-interactive run must never block on transformers' y/N prompt.

    transformers only prompts when ``trust_remote_code is None``; we always
    pass an explicit bool. This pins the installed transformers' behaviour for
    an explicit False: it raises the ValueError we classify, and never calls
    ``input``.
    """

    def test_explicit_false_raises_without_input(self, monkeypatch):
        from transformers.dynamic_module_utils import resolve_trust_remote_code

        def _no_prompt(*_a, **_k):
            raise AssertionError("transformers prompted on stdin")

        monkeypatch.setattr(builtins, "input", _no_prompt)
        with pytest.raises(ValueError) as exc:
            resolve_trust_remote_code(False, "org/custom-model", False, True)
        assert is_trust_remote_code_error(exc.value)

    def test_explicit_false_uses_native_code_when_available(self):
        from transformers.dynamic_module_utils import resolve_trust_remote_code

        # has_local_code (native transformers implementation) + no opt-in: OK.
        assert resolve_trust_remote_code(False, "org/m", True, True) is False
