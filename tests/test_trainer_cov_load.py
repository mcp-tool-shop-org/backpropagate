"""Coverage tests for ``Trainer.load_model`` and the trl trainer factory.

Real objects wherever possible: the transformers + PEFT paths load a REAL tiny
random-weight Llama (``tests/helpers/tiny_models.py``) saved to a local
directory, so ``AutoModelForCausalLM`` / ``AutoTokenizer`` / ``peft`` all run
for real on CPU with no download.

Mock boundary (named per test): ``unsloth`` (CUDA-only optional library, a fake
module in ``sys.modules`` whose ``from_pretrained`` hands back the real tiny
model), ``torchao`` (fake module), the HF loader where a failure that a local
directory cannot produce is needed (401/RuntimeError), and old-PEFT
``LoraConfig`` rejection.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate import trainer as T  # noqa: E402
from backpropagate.exceptions import (  # noqa: E402
    GPUNotAvailableError,
    ModelLoadError,
    TrainingError,
    TrustRemoteCodeRequiredError,
)
from backpropagate.trainer import Trainer  # noqa: E402
from tests.helpers.tiny_models import tiny_gpt2, tiny_llama, tiny_tokenizer  # noqa: E402

LOGGER = "backpropagate.trainer"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_model")
    tiny_llama(layers=2).save_pretrained(d)
    tiny_tokenizer().save_pretrained(d)
    return str(d)


@pytest.fixture(autouse=True)
def _no_unsloth_autoinstall(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")


def make(model, **kw):
    kw.setdefault("use_unsloth", False)
    kw.setdefault("batch_size", 2)
    kw.setdefault("report_to", "none")
    kw.setdefault("max_seq_length", 32)
    return Trainer(model=model, **kw)


def lora_make(model, **kw):
    """LoRA trainer that skips bitsandbytes 4-bit (CUDA-only) -> bf16 base."""
    kw.setdefault("load_in_4bit", False)
    return make(model, **kw)


# ---------------------------------------------------------------------------
# transformers (+ PEFT) path on a real tiny model
# ---------------------------------------------------------------------------

class TestTransformersFullFt:
    def test_full_mode_loads_plain_fp32_trainable_model_on_cpu(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        t.load_model()
        assert t._is_loaded is True
        assert all(p.requires_grad for p in t._model.parameters())
        assert all(p.dtype.is_floating_point and str(p.dtype) == "torch.float32"
                   for p in t._model.parameters())
        assert not hasattr(t._model, "peft_config")
        assert t._tokenizer.pad_token is not None
        first = t._model
        t.load_model()  # idempotent
        assert t._model is first

    def test_fp16_resolution_loads_half_precision_weights(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        with patch.object(Trainer, "_detect_optimal_dtype", return_value=(False, True)):
            t.load_model()
        assert {str(p.dtype) for p in t._model.parameters()} == {"torch.float16"}

    def test_bf16_resolution_loads_bfloat16_weights(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        with patch.object(Trainer, "_detect_optimal_dtype", return_value=(True, False)):
            t.load_model()
        assert {str(p.dtype) for p in t._model.parameters()} == {"torch.bfloat16"}

    def test_full_ceiling_rechecked_against_the_loaded_models_real_param_count(self, tiny_dir):
        from backpropagate.exceptions import FullFinetuneModelTooLargeError

        # The construction gate cannot size a local path, so it defers; the
        # load-time recheck reads the real num_parameters() and refuses.
        t = make(tiny_dir, mode="full", full_ft_ceiling_billions=1e-6)
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            t.load_model()
        assert ei.value.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE"
        assert ei.value.param_count_billions < 1e-3
        assert ei.value.ceiling_billions == pytest.approx(1e-6)
        assert t._is_loaded is True  # the load itself succeeded; the gate refused after


class TestTransformersLora:
    def test_unquantized_lora_attaches_real_adapter(self, tiny_dir):
        t = lora_make(tiny_dir, lora_r=4, lora_alpha=8)
        t.load_model()
        assert hasattr(t._model, "peft_config")
        cfg = t._model.peft_config["default"]
        assert cfg.r == 4 and cfg.lora_alpha == 8
        assert t.lora_adapted_modules > 0
        trainable = [n for n, p in t._model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)

    def test_dora_rslora_and_init_weights_are_forwarded_to_peft(self, tiny_dir):
        t = lora_make(tiny_dir, lora_r=4, use_dora=True, use_rslora=True,
                      init_lora_weights="gaussian")
        t.load_model()
        cfg = t._model.peft_config["default"]
        assert cfg.use_dora is True
        assert cfg.use_rslora is True
        assert cfg.init_lora_weights == "gaussian"

    def test_old_peft_rejecting_new_kwargs_degrades_with_warning(self, tiny_dir, caplog):
        import peft

        real = peft.LoraConfig
        seen = []

        def old_peft(**kw):
            seen.append(set(kw))
            if {"use_dora", "use_rslora", "init_lora_weights"} & set(kw):
                raise TypeError("unexpected keyword argument 'use_dora'")
            return real(**kw)

        t = lora_make(tiny_dir, lora_r=4, use_dora=True, use_rslora=True,
                      init_lora_weights="gaussian")
        with patch("peft.LoraConfig", side_effect=old_peft), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            t.load_model()
        assert "PEFT LoraConfig rejected kwarg(s) ['use_dora', 'use_rslora', 'init_lora_weights']" in caplog.text
        assert len(seen) == 2 and not ({"use_dora", "use_rslora"} & seen[1])
        cfg = t._model.peft_config["default"]
        assert cfg.use_dora is False and cfg.use_rslora is False
        assert t.lora_adapted_modules > 0

    def test_only_present_kwargs_are_reported_as_stripped(self, tiny_dir, caplog):
        import peft

        real = peft.LoraConfig

        def old_peft(**kw):
            if "use_rslora" in kw:
                raise TypeError("no rslora")
            return real(**kw)

        t = lora_make(tiny_dir, lora_r=4, use_rslora=True)
        with patch("peft.LoraConfig", side_effect=old_peft), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            t.load_model()
        assert "rejected kwarg(s) ['use_rslora']" in caplog.text


class TestTransformersLoadFailures:
    def test_tokenizer_load_failure_surfaces_as_model_load_error(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        with patch("transformers.AutoTokenizer.from_pretrained",
                   side_effect=OSError("tokenizer.json not found")):
            with pytest.raises(ModelLoadError) as ei:
                t.load_model()
        assert ei.value.model_name == tiny_dir
        assert "tokenizer.json not found" in ei.value.reason
        assert t._is_loaded is False

    def test_missing_model_directory_raises_model_load_error(self, tmp_path):
        t = make(str(tmp_path / "does-not-exist"), mode="full")
        with pytest.raises(ModelLoadError):
            t.load_model()
        assert t._is_loaded is False

    def test_tokenizer_needing_remote_code_is_structured(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        err = ValueError("requires you to execute code; set trust_remote_code=True")
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=err):
            with pytest.raises(TrustRemoteCodeRequiredError):
                t.load_model()

    def test_pad_token_falls_back_to_eos(self, tiny_dir):
        tok = tiny_tokenizer()
        tok.pad_token = None
        t = make(tiny_dir, mode="full")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok):
            t.load_model()
        assert t._tokenizer.pad_token == t._tokenizer.eos_token

    def test_non_cuda_runtime_error_is_a_classified_model_load_error(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                   side_effect=RuntimeError("size mismatch for embed_tokens")):
            with pytest.raises(ModelLoadError) as ei:
                t.load_model()
        assert ei.value.cause_category == "unknown"
        assert "size mismatch" in ei.value.reason

    def test_cuda_runtime_error_maps_to_gpu_not_available(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                   side_effect=RuntimeError("CUDA driver version is insufficient")):
            with pytest.raises(GPUNotAvailableError):
                t.load_model()

    def test_import_error_maps_to_version_category(self, tiny_dir):
        t = make(tiny_dir, mode="full")
        with patch("transformers.AutoModelForCausalLM.from_pretrained",
                   side_effect=ImportError("cannot import name 'x'")):
            with pytest.raises(ModelLoadError) as ei:
                t.load_model()
        assert ei.value.cause_category == "version"

    def test_hub_auth_error_is_classified(self, tiny_dir):
        import httpx
        from huggingface_hub.utils import HfHubHTTPError

        req = httpx.Request("GET", "https://huggingface.co/x")
        err = HfHubHTTPError("401", response=httpx.Response(401, request=req))
        t = make(tiny_dir, mode="full")
        with patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=err):
            with pytest.raises(ModelLoadError) as ei:
                t.load_model()
        assert ei.value.cause_category == "auth"


# ---------------------------------------------------------------------------
# Unsloth path (fake unsloth, real tiny model)
# ---------------------------------------------------------------------------

class FakeFastLanguageModel:
    """Stand-in for ``unsloth.FastLanguageModel`` returning the real tiny model."""

    calls: dict = {}

    def __init__(self):
        FakeFastLanguageModel.calls = {}

    @staticmethod
    def from_pretrained(**kw):
        FakeFastLanguageModel.calls["from_pretrained"] = kw
        model = tiny_llama(layers=2)
        return model, tiny_tokenizer()

    @staticmethod
    def get_peft_model(model, **kw):
        FakeFastLanguageModel.calls["get_peft_model"] = kw
        return model


@pytest.fixture
def fake_unsloth(monkeypatch):
    FakeFastLanguageModel()  # reset recorded calls
    mod = types.ModuleType("unsloth")
    mod.FastLanguageModel = FakeFastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", mod)
    monkeypatch.setattr(T, "check_feature", lambda name: name == "unsloth")
    return FakeFastLanguageModel


class TestUnslothLoad:
    def test_lora_forwards_all_features_and_expands_all_linear(self, fake_unsloth, tiny_dir):
        t = make(tiny_dir, use_unsloth=True, use_dora=True, use_rslora=True,
                 init_lora_weights="pissa", lora_r=8)
        t.load_model()
        assert t.use_unsloth is True and t._is_loaded
        fp = fake_unsloth.calls["from_pretrained"]
        assert fp["load_in_4bit"] is True and "full_finetuning" not in fp
        gp = fake_unsloth.calls["get_peft_model"]
        assert gp["r"] == 8
        assert gp["use_dora"] is True and gp["use_rslora"] is True
        assert gp["init_lora_weights"] == "pissa"
        assert set(gp["target_modules"]) == {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def test_default_features_are_not_forwarded(self, fake_unsloth, tiny_dir):
        t = make(tiny_dir, use_unsloth=True, init_lora_weights="default", use_dora=False,
                 use_rslora=False)
        t.load_model()
        gp = fake_unsloth.calls["get_peft_model"]
        assert not ({"use_dora", "use_rslora", "init_lora_weights"} & set(gp))

    def test_full_mode_requests_full_finetuning_and_skips_adapter(self, fake_unsloth, tiny_dir):
        t = make(tiny_dir, use_unsloth=True, mode="full")
        t.load_model()
        fp = fake_unsloth.calls["from_pretrained"]
        assert fp["load_in_4bit"] is False and fp["full_finetuning"] is True
        assert "get_peft_model" not in fake_unsloth.calls

    def test_from_pretrained_failure_falls_back_to_transformers(
        self, fake_unsloth, tiny_dir, caplog
    ):
        t = make(tiny_dir, use_unsloth=True, mode="full")
        with patch.object(FakeFastLanguageModel, "from_pretrained",
                          side_effect=OSError("kernel build failed")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            t.load_model()
        assert t.use_unsloth is False
        assert t._is_loaded is True
        assert "Unsloth load failed (ModelLoadError" in caplog.text
        assert "falling back to transformers + PEFT" in caplog.text

    def test_trust_remote_code_refusal_is_not_retried_on_transformers(self, fake_unsloth, tiny_dir):
        t = make(tiny_dir, use_unsloth=True)
        err = ValueError("pass trust_remote_code=True to run custom code")
        with patch.object(FakeFastLanguageModel, "from_pretrained", side_effect=err):
            with pytest.raises(TrustRemoteCodeRequiredError):
                t.load_model()

    def test_model_without_linear_layers_is_refused_before_peft(self, fake_unsloth, tiny_dir):
        import torch

        t = make(tiny_dir, use_unsloth=True, unsloth_fallback=False)
        bare = torch.nn.Sequential(torch.nn.LayerNorm(4))
        with patch.object(FakeFastLanguageModel, "from_pretrained",
                          return_value=(bare, tiny_tokenizer())):
            with pytest.raises(ModelLoadError) as ei:
                t.load_model()
        assert "matched no linear layers" in ei.value.reason
        assert "get_peft_model" not in fake_unsloth.calls

    def test_fp8_effective_forces_transformers_backend(self, fake_unsloth, tiny_dir, monkeypatch, caplog):
        from tests.test_trainer_cov_init import _fake_torchao

        calls, _ = _fake_torchao(monkeypatch)
        t = lora_make(tiny_dir, use_unsloth=True, lora_r=4)
        t._fp8_effective = True
        with caplog.at_level(logging.INFO, logger=LOGGER):
            t.load_model()
        assert t.use_unsloth is False
        assert "from_pretrained" not in fake_unsloth.calls  # unsloth never touched
        assert "fp8: forcing the transformers backend" in caplog.text
        assert calls  # torchao conversion ran after the LoRA attach
        assert t._fp8_effective is True


# ---------------------------------------------------------------------------
# target-module helpers
# ---------------------------------------------------------------------------

class TestTargetModuleHelpers:
    def test_llama_all_linear_leaf_names_exclude_lm_head(self):
        assert T._all_linear_leaf_names(tiny_llama(layers=1)) == [
            "down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]

    def test_gpt2_conv1d_layers_are_included(self):
        names = T._all_linear_leaf_names(tiny_gpt2(layers=1))
        assert {"c_attn", "c_proj", "c_fc"} <= set(names)

    def test_without_conv1d_support_only_linear_is_considered(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers.pytorch_utils", None)
        names = T._all_linear_leaf_names(tiny_gpt2(layers=1))
        assert "c_attn" not in names

    def test_model_without_output_embedding_still_lists_linears(self):
        import torch

        class NoHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

            def get_output_embeddings(self):
                raise NotImplementedError

        assert T._all_linear_leaf_names(NoHead()) == ["proj"]

    def test_model_returning_no_output_embedding(self):
        import torch

        class NoneHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

            def get_output_embeddings(self):
                return None

        assert T._all_linear_leaf_names(NoneHead()) == ["fc"]

    @pytest.mark.parametrize("value,expected", [
        ("q_proj", ["q_proj"]),
        (("q_proj", "v_proj"), ["q_proj", "v_proj"]),
        (["a", "b"], ["a", "b"]),
    ])
    def test_unsloth_target_modules_shaping(self, value, expected):
        assert T._unsloth_target_modules(value, object()) == expected

    def test_all_linear_is_case_insensitive(self):
        assert "q_proj" in T._unsloth_target_modules("ALL-LINEAR", tiny_llama(layers=1))

    def test_count_lora_layers_counts_adapters_and_tolerates_junk(self, tiny_dir):
        t = lora_make(tiny_dir, lora_r=4)
        t.load_model()
        assert T._count_lora_layers(t._model) == t.lora_adapted_modules > 0
        assert T._count_lora_layers(tiny_llama(layers=1)) == 0
        assert T._count_lora_layers(object()) == 0  # no named_modules -> 0


# ---------------------------------------------------------------------------
# trl trainer factory (preference objectives)
# ---------------------------------------------------------------------------

class TestBuildTrainerPreferenceResolution:
    def _trainer(self, method, **kw):
        t = make("acme/Tiny-1B", method=method, **kw)
        t._model = types.SimpleNamespace()
        t._tokenizer = object()
        return t

    @pytest.mark.parametrize("method,name,exp_mod", [
        ("orpo", "ORPOTrainer", "trl.experimental.orpo"),
        ("simpo", "CPOTrainer", "trl.experimental.cpo"),
        ("kto", "KTOTrainer", "trl.experimental.kto"),
    ])
    def test_falls_back_to_experimental_module_when_top_level_lacks_class(
        self, monkeypatch, method, name, exp_mod
    ):
        built = {}

        class Fake:
            def __init__(self, **kw):
                built.update(kw)

        stub_top = types.ModuleType("trl")  # no trainers at top level
        stub_exp = types.ModuleType(exp_mod)
        setattr(stub_exp, name, Fake)
        monkeypatch.setitem(sys.modules, "trl", stub_top)
        monkeypatch.setitem(sys.modules, exp_mod, stub_exp)
        t = self._trainer(method)
        out = t._build_trainer("ARGS", "DATA", ["cb"])
        assert isinstance(out, Fake)
        assert built["args"] == "ARGS" and built["train_dataset"] == "DATA"
        assert built["callbacks"] == ["cb"]
        assert "ref_model" not in built  # reference-free / adapter-as-reference
        # transformers-5 shim: inert warnings_issued dict provided
        assert t._model.warnings_issued == {}

    def test_both_locations_missing_raises_structured_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "trl", types.ModuleType("trl"))
        monkeypatch.setitem(sys.modules, "trl.experimental.kto", types.ModuleType("trl.experimental.kto"))
        t = self._trainer("kto")
        with pytest.raises(TrainingError) as ei:
            t._build_trainer("A", "D", None)
        assert ei.value.code == "RUNTIME_TRAINING_FAILED"
        assert "KTO objective is unavailable" in str(ei.value)
