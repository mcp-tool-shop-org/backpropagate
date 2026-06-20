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

    @pytest.mark.parametrize(
        "vram,expected",
        [(None, 4.0), (16, 4.0), (24, 7.0), (32, 8.0), (48, 16.0), (23.6, 7.0), (31.8, 8.0)],
    )
    def test_offload_ceiling(self, vram, expected):
        assert t._full_ft_offload_ceiling_for_vram(vram) == expected

    def test_offload_ceiling_strictly_higher_at_32gb(self):
        # The whole point: offload lifts the 32 GB full-FT ceiling past 7B.
        assert t._full_ft_offload_ceiling_for_vram(32) > t._full_ft_ceiling_for_vram(32)
        assert t._full_ft_offload_ceiling_for_vram(32) >= 7.0


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
                offload_ceiling_billions=t._full_ft_offload_ceiling_for_vram(32),
                full_ft_offload=False,
            )
        msg = str(ei.value)
        assert "--full-ft-offload" in msg
        assert ei.value.offload_recoverable is True

    def test_7b_offload_32gb_approved(self):
        """With offload on, 7B clears the 8B offload ceiling -> no raise."""
        t._enforce_full_ft_param_ceiling(
            self.SEVEN_B,
            ceiling_billions=t._full_ft_offload_ceiling_for_vram(32),
            offload_ceiling_billions=t._full_ft_offload_ceiling_for_vram(32),
            full_ft_offload=True,
        )

    def test_70b_exceeds_even_offload_names_lora(self):
        """A 70B model exceeds even the offload ceiling -> recovery is LoRA/QLoRA,
        NOT --full-ft-offload."""
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            t._enforce_full_ft_param_ceiling(
                "meta-llama/Llama-3.1-70B-Instruct",
                ceiling_billions=t._full_ft_offload_ceiling_for_vram(32),
                offload_ceiling_billions=t._full_ft_offload_ceiling_for_vram(32),
                full_ft_offload=True,
            )
        msg = str(ei.value).lower()
        assert "lora" in msg
        assert ei.value.offload_recoverable is False

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
