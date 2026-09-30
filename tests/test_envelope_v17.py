"""v1.7 "32 GB envelope" tests.

Covers the card-aware full-FT ceiling, the FSDP2 CPU-offload gate + contrastive
recovery, the DEP_FSDP_UNAVAILABLE runtime guard, the 32/48 GB batch tiers, the
estimate_vram offload modeling, the 24-34B QLoRA presets, and the MLX
unverified-preview reframe. All unit/mocked — the real-GPU FSDP smoke is run
separately by the coordinator (it needs NCCL / WSL2).
"""

from __future__ import annotations

import pytest

import backpropagate.trainer as t
from backpropagate.exceptions import (
    ERROR_CODES,
    FsdpUnavailableError,
    FullFinetuneModelTooLargeError,
)


# ---------------------------------------------------------------------------
# Card-aware full-FT ceilings (4-addend arithmetic, measured anchors)
# ---------------------------------------------------------------------------
class TestFullFtCeilingAnchors:
    @pytest.mark.parametrize(
        "vram,expected",
        [
            (None, 4.0), (8, 4.0), (16, 4.0), (24, 5.0), (32, 6.0), (48, 10.0), (80, 10.0),
            # Realistic *reported* total_memory (nominal-vs-reported tolerance):
            # a "32 GB" 5090 reports ~31.8 GiB. Regression for the GPU-smoke bug
            # where 31.8 fell into the 24 GB tier and resolved 5.0B not 6.0B.
            (15.5, 4.0), (23.6, 5.0), (31.8, 6.0), (47.5, 10.0),
        ],
    )
    def test_pure_gpu_ceiling(self, vram, expected):
        assert t._full_ft_ceiling_for_vram(vram) == expected

    # The v1.7 offload VRAM table (24 GB -> 7B, 32 GB -> 8B) is gone. It was
    # never measured and never looked at host RAM. The offload ceiling now comes
    # from AVAILABLE host RAM through the measured model; the full RAM + VRAM
    # check is in tests/test_offload_fit.py.
    @pytest.mark.parametrize(
        "available_gib,expected",
        [(64.0, 14.46), (28.0, 4.80), (10.0, 0.0)],
    )
    def test_offload_ceiling_from_host_ram(self, monkeypatch, available_gib, expected):
        import backpropagate.offload_engine as oe

        monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: (available_gib, available_gib))
        assert t._full_ft_offload_ceiling_billions() == pytest.approx(expected, abs=0.01)

    def test_offload_ceiling_falls_back_when_ram_unknown(self, monkeypatch):
        import backpropagate.offload_engine as oe

        monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: (None, None))
        assert t._full_ft_offload_ceiling_billions() == t._FULL_FT_PARAM_CEILING_BILLIONS


# ---------------------------------------------------------------------------
# Ceiling gate + contrastive recovery
# ---------------------------------------------------------------------------
class TestCeilingGate:
    SEVEN_B = "Qwen/Qwen2.5-7B-Instruct"

    def test_7b_pure_gpu_32gb_rejected_names_offload(self):
        """7B full-FT on a 32 GB card without offload exceeds the 6B pure-GPU
        ceiling -> raise, naming --full-ft-offload as the contrastive recovery."""
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            t._enforce_full_ft_param_ceiling(
                self.SEVEN_B,
                ceiling_billions=t._full_ft_ceiling_for_vram(32),
                offload_ceiling_billions=14.46,  # 64 GiB available (measured model)
                full_ft_offload=False,
            )
        msg = str(ei.value)
        assert "--full-ft-offload" in msg
        assert ei.value.offload_recoverable is True

    def test_offload_is_not_gated_by_a_param_table(self):
        """With offload on, the param-count table no longer gates (effective
        ceiling = inf); the measured fit check does (tests/test_offload_fit.py)."""
        t._enforce_full_ft_param_ceiling(
            "meta-llama/Llama-3.1-70B-Instruct",
            ceiling_billions=float("inf"),
            offload_ceiling_billions=14.46,
            full_ft_offload=True,
        )

    def test_70b_offload_fails_the_fit_check_naming_lora(self):
        """A 70B model fails the measured host-RAM check even at 64 GiB; the
        recovery names LoRA/QLoRA and not --full-ft-offload."""
        from backpropagate.exceptions import OffloadDoesNotFitError
        from backpropagate.offload_engine import check_offload_fit

        report = check_offload_fit(
            params=70.6e9, root_unit_bytes=None, layer_bytes=None, tokens=512,
            host_total_gib=64.0, host_available_gib=64.0, vram_total_gib=31.4,
        )
        assert report["fits"] is False
        err = OffloadDoesNotFitError("meta-llama/Llama-3.1-70B-Instruct", report)
        assert "lora" in str(err).lower()
        assert err.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE"
        assert isinstance(err, FullFinetuneModelTooLargeError)

    def test_explicit_ceiling_override_allows_7b_without_offload(self):
        """--full-ft-ceiling-billions raises the ceiling so 7B passes pure-GPU."""
        t._enforce_full_ft_param_ceiling(
            self.SEVEN_B,
            ceiling_billions=8.0,  # operator override
            offload_ceiling_billions=None,
            full_ft_offload=False,
        )


# ---------------------------------------------------------------------------
# DEP_FSDP_UNAVAILABLE
# ---------------------------------------------------------------------------
class TestDepFsdp:
    def test_code_registered(self):
        assert "DEP_FSDP_UNAVAILABLE" in ERROR_CODES
        entry = ERROR_CODES["DEP_FSDP_UNAVAILABLE"]
        assert entry["description"]
        assert entry["default_hint"]

    def test_error_carries_code(self):
        err = FsdpUnavailableError("test reason")
        assert err.code == "DEP_FSDP_UNAVAILABLE"
        assert "test reason" in str(err)

    def test_runtime_guard_raises_without_nccl(self, monkeypatch):
        """On a host without NCCL (e.g. Windows-native), the offload runtime
        guard fails fast with DEP_FSDP_UNAVAILABLE naming WSL2/Linux."""
        import torch
        import torch.distributed as dist

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
        with pytest.raises(FsdpUnavailableError) as ei:
            t._ensure_fsdp_runtime()
        assert "NCCL" in str(ei.value)


# ---------------------------------------------------------------------------
# FSDP2 offload SFTConfig — CPU pins for the bugs the real-GPU smoke found
# (tests/test_full_ft_offload_smoke.py is the end-to-end regression test; these
# keep the config contract pinned in CI, where the smoke cannot run).
# ---------------------------------------------------------------------------
class TestOffloadSftConfig:
    @staticmethod
    def _build(**overrides):
        from unittest.mock import patch

        # Resolve trl's lazy SFTConfig BEFORE the CUDA mocks: importing it under
        # a MagicMock device-props patch blows up inside trl's import.
        import trl

        _ = trl.SFTConfig
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties") as props, \
             patch("torch.cuda.get_device_capability", return_value=(12, 0)), \
             patch("trl.SFTConfig") as sft:
            props.return_value.total_memory = 32 * 1024 ** 3
            kwargs = {
                "output_dir": "/tmp/out",
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "max_steps": 2,
                "learning_rate": 2e-5,
                "warmup_steps": 0,
                "max_seq_length": 128,
                "seed": 42,
                "lr_scheduler_type": "cosine",
                "logging_steps": 1,
                "mode": "full",
            }
            kwargs.update(overrides)
            t._build_sft_config(**kwargs)
            return sft.call_args.kwargs

    def test_offload_disables_trainingargs_gradient_checkpointing_explicitly(self):
        """TRL's SFTConfig defaults gradient_checkpointing=True, so popping the
        key re-enabled it and transformers refused the FSDP config."""
        kw = self._build(full_ft_offload=True)
        assert kw["gradient_checkpointing"] is False
        assert "gradient_checkpointing_kwargs" not in kw
        assert kw["fsdp_config"]["activation_checkpointing"] is True

    @pytest.mark.parametrize("pinned", [None, "adamw_8bit", "paged_adamw_8bit"])
    def test_offload_never_uses_a_bitsandbytes_optimizer(self, pinned):
        """bnb optimizers cannot step CPU-offloaded DTensor params."""
        kw = self._build(full_ft_offload=True, optim=pinned)
        assert kw["optim"] == "adamw_torch"

    def test_offload_honors_a_torch_optimizer_pin(self):
        kw = self._build(full_ft_offload=True, optim="adamw_torch_fused")
        assert kw["optim"] == "adamw_torch_fused"

    def test_offload_checkpoints_are_sharded(self):
        """A single-process FULL_STATE_DICT checkpoint gathers the whole model +
        optimizer onto the GPU; SHARDED_STATE_DICT writes the CPU shards."""
        kw = self._build(full_ft_offload=True)
        assert kw["fsdp_config"]["state_dict_type"] == "SHARDED_STATE_DICT"

    def test_pure_gpu_full_ft_unchanged(self):
        kw = self._build(full_ft_offload=False)
        assert kw["gradient_checkpointing"] is True
        assert kw["optim"] == "paged_adamw_8bit"
        assert "fsdp" not in kw

    def test_gather_full_state_dict_is_noop_for_unsharded_models(self):
        import torch

        assert t._gather_fsdp_full_state_dict(torch.nn.Linear(2, 2)) is None


# ---------------------------------------------------------------------------
# 32/48 GB batch tiers
# ---------------------------------------------------------------------------
class TestBatchTiers:
    @pytest.mark.parametrize(
        "vram,expected",
        [
            (80, 8), (48, 8), (32, 6), (24, 4), (16, 2), (12, 1), (8, 1),
            # Realistic reported total_memory (tolerance) — a 5090 reports ~31.8.
            (47.5, 8), (31.8, 6), (23.6, 4), (15.5, 2),
        ],
    )
    def test_detect_batch_size_tiers(self, vram, expected, monkeypatch):
        import torch

        class _Props:
            total_memory = int(vram * (1024 ** 3))

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda idx: _Props()
        )
        tr = t.Trainer.__new__(t.Trainer)  # avoid heavy __init__
        assert tr._detect_batch_size() == expected

    def test_cli_tier_table_mirrors(self):
        from backpropagate.cli import _VRAM_BATCH_SIZE_TIERS

        tiers = {threshold: bs for threshold, bs, _ in _VRAM_BATCH_SIZE_TIERS}
        assert tiers[48.0] == 8
        assert tiers[32.0] == 6
        assert tiers[24.0] == 4


# ---------------------------------------------------------------------------
# estimate_vram offload modeling
# ---------------------------------------------------------------------------
class TestEstimateVramOffload:
    SEVEN_B = "Qwen/Qwen2.5-7B-Instruct"

    def test_offload_sets_host_ram_and_shrinks_gpu(self):
        on_gpu = t.estimate_vram(self.SEVEN_B, mode="full", offload=False)
        offloaded = t.estimate_vram(self.SEVEN_B, mode="full", offload=True)
        assert on_gpu.host_ram_gb == 0.0
        assert offloaded.host_ram_gb > 0.0
        # Offloading params+optimizer to host shrinks the GPU footprint a lot.
        assert offloaded.total_gb < on_gpu.total_gb
        # 7B host spill is on the order of tens of GB (fits 64 GB host RAM).
        assert 20 < offloaded.host_ram_gb < 64

    def test_vramestimate_has_host_ram_field(self):
        est = t.estimate_vram(self.SEVEN_B, mode="full", offload=True)
        assert hasattr(est, "host_ram_gb")
        assert "host_ram" in est.summary()


# ---------------------------------------------------------------------------
# 24-34B QLoRA presets
# ---------------------------------------------------------------------------
class TestEnvelopePresets:
    @pytest.mark.parametrize(
        "name", ["llama-3.1-8b", "qwen2.5-14b", "mistral-small-24b", "qwen2.5-32b"]
    )
    def test_preset_present(self, name):
        from backpropagate.config import MODEL_PRESETS

        assert name in MODEL_PRESETS

    def test_presets_resolve_by_id(self):
        from backpropagate.config import lookup_model_preset_by_id

        assert lookup_model_preset_by_id("Qwen/Qwen2.5-32B-Instruct") is not None


# ---------------------------------------------------------------------------
# MLX reframed to unverified preview
# ---------------------------------------------------------------------------
class TestMlxReframe:
    def test_feature_description_is_preview(self):
        from backpropagate.feature_flags import FEATURE_DESCRIPTIONS

        desc = FEATURE_DESCRIPTIONS["mlx"].lower()
        assert any(w in desc for w in ("preview", "experimental", "unverified"))

    def test_no_version_hype_in_mlx_backend_source(self):
        from pathlib import Path

        import backpropagate.mlx_backend as mb

        src = Path(mb.__file__).read_text(encoding="utf-8").lower()
        assert "new in v1.5" not in src
