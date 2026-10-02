"""Coverage tests for ``Trainer.__init__`` validation/gates and the hardware
resolvers (batch size, optimizer, dtype, FP8 support + conversion, Windows env).

Mock boundary: CUDA / ``torch.cuda`` probes (hardware), ``torchao`` (optional
GPU-only library, replaced by a fake module in ``sys.modules``), the host RAM
probe in ``offload_engine`` and HF ``AutoConfig`` (network). Everything else is
the real Trainer code on real objects (no model is loaded here).
"""

from __future__ import annotations

import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backpropagate import trainer as T
from backpropagate.exceptions import (
    InvalidSettingError,
    TrainingError,
)
from backpropagate.trainer import Trainer

LOGGER = "backpropagate.trainer"
TINY_ID = "acme/Tiny-1B"  # not a preset; 1B by the id heuristic


def make(**kw):
    kw.setdefault("model", TINY_ID)
    kw.setdefault("use_unsloth", False)
    kw.setdefault("batch_size", 2)
    kw.setdefault("report_to", "none")
    return Trainer(**kw)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    @pytest.mark.parametrize("kw,name", [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -3}, "batch_size"),
        ({"gradient_accumulation": 0}, "gradient_accumulation"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": -1e-4}, "learning_rate"),
        ({"lora_r": 0}, "lora_r"),
        ({"lora_alpha": 0}, "lora_alpha"),
        ({"lora_dropout": 1.5}, "lora_dropout"),
        ({"lora_dropout": -0.1}, "lora_dropout"),
        ({"max_seq_length": 0}, "max_seq_length"),
        ({"mode": "turbo"}, "mode"),
        ({"method": "dpo"}, "method"),
        ({"backend": "rocm"}, "backend"),
    ])
    def test_invalid_knob_raises_structured_error_naming_it(self, kw, name):
        with pytest.raises(InvalidSettingError) as ei:
            make(**kw)
        assert ei.value.code == "CONFIG_INVALID_SETTING"
        assert ei.value.setting_name == name
        assert ei.value.suggestion  # every refusal carries a remedy

    def test_zero_dropout_and_unit_dropout_are_legal_boundaries(self):
        assert make(lora_dropout=0.0).lora_dropout == 0.0
        assert make(lora_dropout=1.0).lora_dropout == 1.0

    @pytest.mark.parametrize("kw,setting", [
        ({"method": "orpo", "orpo_beta": 0.0}, "orpo_beta"),
        ({"method": "orpo", "orpo_beta": -1.0}, "orpo_beta"),
        ({"method": "simpo", "simpo_gamma": 0.0}, "simpo_gamma"),
        ({"method": "kto", "kto_beta": 0.0}, "kto_beta"),
        ({"method": "kto", "kto_desirable_weight": -1.0}, "kto_desirable_weight"),
        ({"method": "kto", "kto_undesirable_weight": 0.0}, "kto_undesirable_weight"),
    ])
    def test_objective_specific_values_refused(self, kw, setting):
        with pytest.raises(InvalidSettingError) as ei:
            make(**kw)
        assert ei.value.setting_name == setting

    def test_preference_knobs_are_inert_for_other_methods(self):
        t = make(method="sft", orpo_beta=-5.0, simpo_gamma=-1.0, kto_beta=0.0)
        assert t.method == "sft"

    @pytest.mark.parametrize("method", ["orpo", "simpo", "kto"])
    def test_preference_methods_refuse_full_mode(self, method):
        with pytest.raises(InvalidSettingError) as ei:
            make(method=method, mode="full")
        assert ei.value.setting_name == "method+mode"

    def test_default_batch_is_resolved_from_the_card(self):
        with patch("torch.cuda.is_available", return_value=False):
            t = Trainer(model=TINY_ID, use_unsloth=False, report_to="none")
        assert t.batch_size == 2  # CPU fallback

    def test_auto_optim_sentinel_uses_settings_default(self):
        from backpropagate.config import settings

        assert make(optim="auto").optim == settings.training.optim
        assert make(optim="adamw_torch").optim == "adamw_torch"

    def test_lora_preset_defaults_to_quality(self):
        assert make().lora_preset == "quality"
        assert make(lora_preset="fast").lora_preset == "fast"


class TestOffloadFlagAndSimpoWarnings:
    def test_offload_flag_inert_for_lora_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            t = make(mode="lora", full_ft_offload=True)
        assert t.full_ft_offload is True
        assert "full_ft_offload=True has no effect with mode='lora'" in caplog.text

    def test_simpo_gamma_over_beta_warns_with_ratio(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            make(method="simpo", simpo_beta=0.5, simpo_gamma=2.0, learning_rate=1e-6)
        assert "simpo_gamma (2.0) / simpo_beta (0.5) = 4.000 > 1.0" in caplog.text

    def test_simpo_ratio_at_or_below_one_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            make(method="simpo", simpo_beta=2.0, simpo_gamma=1.0, learning_rate=1e-6)
        assert "simpo_gamma" not in caplog.text


class TestEngineKnobs:
    def test_stray_block_knobs_are_ignored_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            t = make(mode="full", switch_block_every=10, block_order="ascending")
        assert t.full_ft_engine == "default"
        assert "block_order, switch_block_every only apply with full_ft_engine='block'" in caplog.text

    def test_block_engine_disables_unsloth_for_the_run(self, caplog):
        with patch.object(T, "check_feature", return_value=True), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            t = Trainer(model=TINY_ID, mode="full", use_unsloth=True, batch_size=2,
                        full_ft_engine="block", report_to="none")
        assert t.use_unsloth is False
        assert "using the transformers backend" in caplog.text
        assert "train_on_responses_only masking" in caplog.text


# ---------------------------------------------------------------------------
# License caveat + Windows env + batch-size detection
# ---------------------------------------------------------------------------

class TestLicenseCaveat:
    def test_restricted_preset_logs_license_warning(self, caplog):
        from backpropagate.config import MODEL_PRESETS

        restricted = next(
            (p for p in MODEL_PRESETS.values() if p.license_restriction), None)
        if restricted is None:
            pytest.skip("no preset in the catalog carries a license restriction")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            make(model=restricted.model_id)
        assert restricted.license_restriction in caplog.text
        assert f"preset={restricted.name}" in caplog.text

    def test_lookup_failure_never_blocks_construction(self, caplog):
        with patch("backpropagate.config.lookup_model_preset_by_id",
                   side_effect=RuntimeError("catalog broken")), \
                caplog.at_level(logging.DEBUG, logger=LOGGER):
            t = make()
        assert t.model_name == TINY_ID
        assert "license-restriction check skipped" in caplog.text


class TestWindowsFixes:
    def test_windows_sets_env_and_honors_settings_flags(self, monkeypatch):
        t = make()
        for k in ("TOKENIZERS_PARALLELISM", "HF_HUB_ENABLE_HF_TRANSFER",
                  "XFORMERS_DISABLED", "CUDA_LAUNCH_BLOCKING"):
            monkeypatch.setenv(k, "unset-marker")
        with patch("backpropagate.trainer.os.name", "nt"), \
                patch("backpropagate.trainer.settings.windows.xformers_disabled", True), \
                patch("backpropagate.trainer.settings.windows.cuda_launch_blocking", True):
            t._apply_windows_fixes()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
        assert os.environ["XFORMERS_DISABLED"] == "1"
        assert os.environ["CUDA_LAUNCH_BLOCKING"] == "1"

    def test_windows_leaves_optional_flags_alone_when_settings_off(self, monkeypatch):
        t = make()  # construction itself applies the fixes on a real Windows host
        monkeypatch.setenv("XFORMERS_DISABLED", "keep")
        monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "keep")
        with patch("backpropagate.trainer.os.name", "nt"), \
                patch("backpropagate.trainer.settings.windows.xformers_disabled", False), \
                patch("backpropagate.trainer.settings.windows.cuda_launch_blocking", False):
            t._apply_windows_fixes()
        assert os.environ["XFORMERS_DISABLED"] == "keep"
        assert os.environ["CUDA_LAUNCH_BLOCKING"] == "keep"

    def test_posix_touches_nothing(self, monkeypatch):
        t = make()
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "keep")
        with patch("backpropagate.trainer.os.name", "posix"):
            t._apply_windows_fixes()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "keep"


class TestDetectBatchSize:
    def _t(self):
        return make()

    @pytest.mark.parametrize("vram_gb,expected", [
        (80, 8), (46.6, 8), (31.8, 6), (24, 4), (23.6, 4), (16, 2), (15.5, 2),
        (12, 1), (11, 1), (8, 1),
    ])
    def test_vram_tiers(self, vram_gb, expected):
        props = SimpleNamespace(total_memory=int(vram_gb * 1024**3))
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_properties", return_value=props):
            assert self._t()._detect_batch_size() == expected

    def test_no_cuda_defaults_to_two(self, caplog):
        with patch("torch.cuda.is_available", return_value=False), \
                caplog.at_level(logging.INFO, logger=LOGGER):
            assert self._t()._detect_batch_size() == 2
        assert "reason=cuda_not_available" in caplog.text

    def test_torch_missing_defaults_to_two(self, monkeypatch, caplog):
        t = self._t()
        monkeypatch.setitem(sys.modules, "torch", None)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert t._detect_batch_size() == 2
        assert "reason=torch_not_installed" in caplog.text

    def test_cuda_runtime_error_defaults_to_two(self, caplog):
        with patch("torch.cuda.is_available", side_effect=RuntimeError("driver hung")), \
                caplog.at_level(logging.INFO, logger=LOGGER):
            assert self._t()._detect_batch_size() == 2
        assert "reason=cuda_query_failed: driver hung" in caplog.text

    def test_unexpected_error_defaults_to_two_with_warning(self, caplog):
        with patch("torch.cuda.is_available", side_effect=ValueError("weird")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            assert self._t()._detect_batch_size() == 2
        assert "reason=unexpected_error: ValueError: weird" in caplog.text


# ---------------------------------------------------------------------------
# Offload fit helpers
# ---------------------------------------------------------------------------

class TestOffloadFitForModel:
    def _patched(self, monkeypatch, *, ram=(128.0, 96.0), vram=32.0):
        from backpropagate import offload_engine as oe

        monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: ram)
        monkeypatch.setattr(T, "_detect_total_vram_gb", lambda: vram)

    def test_meta_probe_failure_falls_back_to_estimated_params(self, monkeypatch, caplog):
        self._patched(monkeypatch)
        t = make(model="acme/Small-1B", mode="full")
        with patch("transformers.AutoConfig.from_pretrained",
                   side_effect=OSError("no network")), \
                caplog.at_level(logging.INFO, logger=LOGGER):
            report = t._enforce_offload_fit_for_model()
        assert report["fits"] is True
        assert "could not build a meta model" in caplog.text
        assert "fit check falls back to the estimated param count" in caplog.text

    def test_meta_probe_failure_with_unknown_size_returns_empty(self, monkeypatch):
        self._patched(monkeypatch)
        t = make(model="acme/mystery-model", mode="full")
        with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")):
            assert t._enforce_offload_fit_for_model() == {}

    def test_does_not_fit_raises_unless_ceiling_override(self, monkeypatch, caplog):
        from backpropagate.exceptions import OffloadDoesNotFitError

        self._patched(monkeypatch, ram=(16.0, 8.0), vram=12.0)
        t = make(model="acme/Big-30B", mode="full", full_ft_ceiling_billions=40.0)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            report = t._enforce_offload_fit(params=30e9)
        assert report["fits"] is False
        assert "overrides it; proceeding" in caplog.text
        t.full_ft_ceiling_billions = None
        with pytest.raises(OffloadDoesNotFitError) as ei:
            t._enforce_offload_fit(params=30e9)
        assert ei.value.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE"

    def test_loaded_model_shape_is_used_when_available(self, monkeypatch):
        from tests.helpers.tiny_models import tiny_llama

        self._patched(monkeypatch)
        t = make(model="acme/Small-1B", mode="full")
        t._model = tiny_llama(layers=2)
        t._is_loaded = True
        with patch("transformers.AutoConfig.from_pretrained",
                   side_effect=AssertionError("must not probe the hub when a model is loaded")):
            report = t._enforce_offload_fit_for_model()
        assert report["fits"] is True


# ---------------------------------------------------------------------------
# Optimizer / dtype resolution
# ---------------------------------------------------------------------------

class TestIsBnbOptim:
    @pytest.mark.parametrize("name,expected", [
        ("", False), (None, False), ("adamw_8bit", True), ("lion_8bit", True),
        ("paged_adamw_32bit", True), ("PAGED_ADAMW", True), ("adamw_torch", False),
        ("sgd", False),
    ])
    def test_predicate(self, name, expected):
        assert Trainer._is_bnb_8bit_optim(name) is expected


class TestDetectOptimForCard:
    def _cuda(self, vram_gb):
        props = SimpleNamespace(total_memory=int(vram_gb * 1024**3))
        return (patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_properties", return_value=props))

    def test_cpu_downgrades_any_bnb_optimizer(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert Trainer._detect_optim_for_card("adamw_8bit") == "adamw_torch"
            assert Trainer._detect_optim_for_card("paged_adamw_8bit") == "adamw_torch"

    def test_non_bnb_optimizer_is_returned_untouched_without_probing(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("must not probe")):
            assert Trainer._detect_optim_for_card("adafactor") == "adafactor"

    def test_cuda_probe_failure_on_bnb_optimizer_falls_back_to_default_resolution(self, caplog):
        # is_available raises -> rule 1 swallowed, rule 3 (second probe) also fails
        # -> the configured optimizer is left unchanged.
        with patch("torch.cuda.is_available", side_effect=RuntimeError("no driver")), \
                caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert Trainer._detect_optim_for_card("paged_adamw_8bit") == "paged_adamw_8bit"
            assert Trainer._detect_optim_for_card("adamw_8bit") == "adamw_8bit"
        assert "CUDA availability probe failed" in caplog.text
        assert "CUDA query failed" in caplog.text

    def test_explicit_non_default_bnb_optimizer_kept_on_cuda(self):
        a, b = self._cuda(16)
        with a, b:
            assert Trainer._detect_optim_for_card("paged_adamw_32bit") == "paged_adamw_32bit"

    def test_default_optim_pages_on_small_cards_and_stays_on_big_cards(self):
        a, b = self._cuda(16)
        with a, b:
            assert Trainer._detect_optim_for_card("adamw_8bit") == "paged_adamw_8bit"
        a, b = self._cuda(32)
        with a, b:
            assert Trainer._detect_optim_for_card("adamw_8bit") == "adamw_8bit"


class TestDetectOptimalDtype:
    def test_explicit_fp16_is_honored(self):
        assert Trainer._detect_optimal_dtype(False, True) == (False, True)
        assert Trainer._detect_optimal_dtype(True, True) == (True, True)

    def test_cpu_forces_fp32(self, caplog):
        with patch("torch.cuda.is_available", return_value=False), \
                caplog.at_level(logging.INFO, logger=LOGGER):
            assert Trainer._detect_optimal_dtype(True, False) == (False, False)
            assert Trainer._detect_optimal_dtype(False, False) == (False, False)
        assert "forcing fp32" in caplog.text

    def test_capability_query_error_leaves_config_unchanged(self, caplog):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", side_effect=RuntimeError("x")), \
                caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert Trainer._detect_optimal_dtype(True, False) == (True, False)
        assert "capability query failed" in caplog.text

    @pytest.mark.parametrize("cap,cfg_bf16,expected", [
        ((8, 0), True, (True, False)),
        ((8, 9), False, (True, False)),     # Ada upgrade from fp16-only config
        ((12, 0), True, (True, False)),
        ((7, 5), True, (False, True)),      # pre-Ampere with bf16 requested
        ((7, 0), False, (False, True)),     # pre-Ampere, nothing requested
    ])
    def test_capability_ladder(self, cap, cfg_bf16, expected, caplog):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=cap), \
                caplog.at_level(logging.INFO, logger=LOGGER):
            assert Trainer._detect_optimal_dtype(cfg_bf16, False) == expected
        if cap == (8, 9) and not cfg_bf16:
            assert "upgrading dtype to bf16" in caplog.text
        if cap == (7, 5):
            assert "pre-Ampere" in caplog.text


# ---------------------------------------------------------------------------
# FP8
# ---------------------------------------------------------------------------

class TestFp8Supported:
    def _t(self):
        return make()

    def test_torch_import_failure(self, monkeypatch):
        t = self._t()
        monkeypatch.setitem(sys.modules, "torch", None)
        ok, reason = t._fp8_supported()
        assert ok is False and "PyTorch is not importable" in reason

    def test_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            ok, reason = self._t()._fp8_supported()
        assert ok is False and "CUDA is not available" in reason

    def test_torchao_missing(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch.object(T, "check_feature", return_value=False):
            ok, reason = self._t()._fp8_supported()
        assert ok is False and "torchao is not installed" in reason

    def test_capability_query_error(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch.object(T, "check_feature", return_value=True), \
                patch("torch.cuda.get_device_capability", side_effect=RuntimeError("bad")):
            ok, reason = self._t()._fp8_supported()
        assert ok is False and "could not query CUDA compute capability" in reason

    def test_ada_names_the_card(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch.object(T, "check_feature", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(8, 9)), \
                patch("torch.cuda.get_device_name", return_value="RTX 4090"):
            ok, reason = self._t()._fp8_supported()
        assert ok is False and "sm_8x (RTX 4090)" in reason

    def test_ada_with_unreadable_name_uses_placeholder(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch.object(T, "check_feature", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
                patch("torch.cuda.get_device_name", side_effect=RuntimeError("nope")):
            ok, reason = self._t()._fp8_supported()
        assert ok is False and "(this GPU)" in reason

    def test_blackwell_is_supported(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch.object(T, "check_feature", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(12, 0)):
            assert self._t()._fp8_supported() == (True, None)


class TestFp8GateLadder:
    def _supported(self):
        return patch.object(Trainer, "_fp8_supported", return_value=(True, None))

    def test_effective_fp8_with_explicit_non_4bit_keeps_flag_and_forces_packing_off(self, caplog):
        with self._supported(), caplog.at_level(logging.INFO, logger=LOGGER):
            t = make(fp8=True, load_in_4bit=False, packing=True)
        assert t._fp8_effective is True
        assert t._load_in_4bit is False
        assert t.packing is False
        assert "fp8: disabling packing (was on)" in caplog.text
        assert "disabling the default 4-bit" not in caplog.text

    def test_effective_fp8_default_4bit_is_flipped_off_with_info(self, caplog):
        with self._supported(), caplog.at_level(logging.INFO, logger=LOGGER):
            t = make(fp8=True, packing=False)
        assert t._load_in_4bit is False
        assert "disabling the default 4-bit base quantization" in caplog.text
        assert "disabling packing" not in caplog.text  # packing was already off

    def test_degrade_keeps_default_4bit_and_does_not_raise(self, caplog):
        # Pin the CPU host instead of relying on the machine running the
        # tests: on a Hopper/Blackwell card with torchao installed the real
        # probe says FP8 is supported and nothing degrades.
        no_cuda = patch("torch.cuda.is_available", return_value=False)
        with no_cuda, caplog.at_level(logging.WARNING, logger=LOGGER):
            t = make(fp8=True)
        assert t._fp8_effective is False
        assert t._load_in_4bit is True
        assert "fp8=True requested but unavailable on this host" in caplog.text
        assert "CUDA is not available" in caplog.text

    @pytest.mark.parametrize("kw,setting", [
        ({"mode": "full"}, "fp8+mode"),
        ({"method": "orpo"}, "fp8+method"),
        ({"load_in_4bit": True}, "fp8+load_in_4bit"),
    ])
    def test_misconfigurations_raise_regardless_of_hardware(self, kw, setting):
        with pytest.raises(InvalidSettingError) as ei:
            make(fp8=True, **kw)
        assert ei.value.setting_name == setting


# ---------------------------------------------------------------------------
# fp8 module filter + conversion
# ---------------------------------------------------------------------------

class TestFp8ModuleFilter:
    @pytest.mark.parametrize("fqn,expected", [
        ("model.layers.0.self_attn.q_proj", True),
        ("model.layers.0.mlp.down_proj", True),
        ("model.layers.0.self_attn.q_proj.lora_A.default", False),
        ("lm_head", False),
        ("model.embed_tokens_proj", False),
        ("model.embedding_proj", False),
    ])
    def test_linear_filtering(self, fqn, expected):
        import torch.nn as nn

        assert Trainer._fp8_module_filter(nn.Linear(16, 16), fqn) is expected

    def test_non_linear_never_converted(self):
        import torch.nn as nn

        assert Trainer._fp8_module_filter(nn.LayerNorm(4), "model.norm") is False


def _fake_torchao(monkeypatch, *, convert=None, f8_count=None):
    """Install a fake ``torchao.float8`` package (GPU-only optional lib)."""
    import torch.nn as nn

    class Float8Linear(nn.Linear):
        pass

    pkg = types.ModuleType("torchao")
    f8 = types.ModuleType("torchao.float8")
    lin = types.ModuleType("torchao.float8.float8_linear")
    lin.Float8Linear = Float8Linear
    calls = {}

    class Float8LinearConfig:
        pass

    def convert_to_float8_training(model, config=None, module_filter_fn=None):
        calls["filter"] = module_filter_fn
        if convert is not None:
            return convert(model, module_filter_fn, Float8Linear)
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and module_filter_fn(mod, name):
                mod.__class__ = Float8Linear
        return model

    f8.Float8LinearConfig = Float8LinearConfig
    f8.convert_to_float8_training = convert_to_float8_training
    f8.float8_linear = lin
    pkg.float8 = f8
    monkeypatch.setitem(sys.modules, "torchao", pkg)
    monkeypatch.setitem(sys.modules, "torchao.float8", f8)
    monkeypatch.setitem(sys.modules, "torchao.float8.float8_linear", lin)
    return calls, Float8Linear


class _Net:
    """Minimal stand-in model exposing ``named_modules`` over real nn.Modules."""

    def __init__(self):
        import torch.nn as nn

        self.mods = [("proj", nn.Linear(16, 16)), ("lm_head", nn.Linear(16, 16)),
                     ("proj.lora_A", nn.Linear(16, 4))]

    def named_modules(self):
        return iter(self.mods)


class TestApplyFp8ToBase:
    def _t(self, effective=True):
        t = make()
        t._fp8_effective = effective
        t._model = _Net()
        return t

    def test_noop_when_not_effective(self):
        t = self._t(effective=False)
        t._apply_fp8_to_base()  # nothing imported, nothing raised
        assert t._fp8_effective is False

    def test_broken_install_raises_structured_runtime_error(self, monkeypatch):
        t = self._t()
        monkeypatch.setitem(sys.modules, "torchao.float8", None)
        monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
        with pytest.raises(TrainingError) as ei:
            t._apply_fp8_to_base()
        assert ei.value.code == "RUNTIME_FP8_UNSUPPORTED"
        assert t._fp8_effective is False

    def test_conversion_converts_base_linears_only(self, monkeypatch, caplog):
        calls, F8 = _fake_torchao(monkeypatch)
        t = self._t()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            t._apply_fp8_to_base()
        assert t._fp8_effective is True
        assert calls["filter"] == Trainer._fp8_module_filter
        kinds = {n: type(m).__name__ for n, m in t._model.named_modules()}
        assert kinds["proj"] == F8.__name__
        assert kinds["lm_head"] == "Linear" and kinds["proj.lora_A"] == "Linear"
        assert "fp8: converted 1 base projection linear(s)" in caplog.text

    def test_zero_matches_degrades_to_bf16(self, monkeypatch, caplog):
        _fake_torchao(monkeypatch, convert=lambda model, flt, F8: model)
        t = self._t()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            t._apply_fp8_to_base()
        assert t._fp8_effective is False
        assert "matched 0 base linears" in caplog.text

    def test_conversion_error_degrades_to_bf16(self, monkeypatch, caplog):
        def boom(model, flt, F8):
            raise RuntimeError("float8 kernel mismatch")

        _fake_torchao(monkeypatch, convert=boom)
        t = self._t()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            t._apply_fp8_to_base()
        assert t._fp8_effective is False
        assert "conversion to Float8Linear failed (RuntimeError: float8 kernel mismatch)" in caplog.text

    def test_counting_failure_still_counts_as_converted(self, monkeypatch, caplog):
        _fake_torchao(monkeypatch)
        t = self._t()
        # named_modules works for the conversion but not for the count: first call
        # ok, second call raises.
        real = t._model.named_modules
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] > 1:
                raise RuntimeError("module tree changed")
            return real()

        t._model.named_modules = flaky
        with caplog.at_level(logging.INFO, logger=LOGGER):
            t._apply_fp8_to_base()
        assert t._fp8_effective is True
        assert "converted an unknown number of base projection linear(s)" in caplog.text


class TestSftConfigFp8Constraints:
    def test_set_shape_constraints_on_a_config_with_all_fields(self):
        t = make()
        cfg = SimpleNamespace(pad_to_multiple_of=None, padding_free=True, packing=True)
        t._set_fp8_shape_constraints_on_sft_config(cfg)
        assert cfg.pad_to_multiple_of == 16
        assert cfg.padding_free is False
        assert cfg.packing is False

    def test_set_shape_constraints_skips_missing_fields(self):
        t = make()
        cfg = SimpleNamespace(pad_to_multiple_of=None)
        t._set_fp8_shape_constraints_on_sft_config(cfg)
        assert cfg.pad_to_multiple_of == 16
        assert not hasattr(cfg, "padding_free") and not hasattr(cfg, "packing")


class TestBuildTrainingArgsFp8:
    def test_effective_fp8_layers_shape_constraints_on_a_real_sft_config(self, tmp_path):
        t = make(output_dir=str(tmp_path / "o"))
        t._fp8_effective = True
        cfg = t._build_training_args(steps=2, report_to="none", run_name=None)
        assert type(cfg).__name__ == "SFTConfig"
        assert cfg.pad_to_multiple_of == 16
        assert cfg.padding_free is False
        assert cfg.packing is False

    def test_non_fp8_leaves_shape_fields_at_trl_defaults(self, tmp_path):
        t = make(output_dir=str(tmp_path / "o"))
        cfg = t._build_training_args(steps=2, report_to="none", run_name=None)
        assert cfg.pad_to_multiple_of != 16
