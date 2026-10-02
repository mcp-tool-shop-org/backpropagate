# Rectangular batches when no variable-length attention kernel exists.
"""``_keep_batches_rectangular`` (trainer.py).

Padding-free / packed training flattens a batch into one sequence of
``batch x seq`` tokens. Without flash-attention or xFormers that is attended
with a dense ``heads x (batch*seq)^2`` mask, so memory grows with the square
of the batch size (measured: a 1B QLoRA at batch 4 x 2048 ran out of memory
on a 32 GB card). These pin the rule that switches such runs to ordinary
``(batch, seq)`` batches, and that it leaves flash / xFormers setups alone.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from backpropagate.trainer import _keep_batches_rectangular, _varlen_attention_available


def _model(attn: str = "sdpa") -> SimpleNamespace:
    return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn))


def _cfg(**kw) -> SimpleNamespace:
    base = {"packing": True, "packing_strategy": "bfd", "padding_free": None}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def dispatch(monkeypatch):
    """A stand-in for unsloth.utils.attention_dispatch with both kernels off."""
    mod = SimpleNamespace(HAS_FLASH_ATTENTION=False, HAS_XFORMERS=False)
    monkeypatch.setitem(sys.modules, "unsloth.utils.attention_dispatch", mod)
    return mod


def test_sdpa_with_packing_switches_to_wrapped(dispatch):
    cfg = _cfg()
    assert _keep_batches_rectangular(cfg, _model(), unsloth_loaded=True) is True
    assert cfg.packing is True
    assert cfg.packing_strategy == "wrapped"
    assert cfg.padding_free is False
    assert cfg._unsloth_disable_auto_packing is True


def test_sdpa_without_packing_turns_padding_free_off(dispatch):
    cfg = _cfg(packing=False)
    assert _keep_batches_rectangular(cfg, _model(), unsloth_loaded=True) is True
    assert cfg.padding_free is False  # not None: Unsloth auto-enables on None
    assert cfg.packing_strategy == "bfd"  # untouched; packing is off
    assert not hasattr(cfg, "_unsloth_disable_auto_packing")


@pytest.mark.parametrize("flag", ["HAS_FLASH_ATTENTION", "HAS_XFORMERS"])
def test_unsloth_with_a_varlen_kernel_is_left_alone(dispatch, flag):
    setattr(dispatch, flag, True)
    cfg = _cfg()
    assert _keep_batches_rectangular(cfg, _model(), unsloth_loaded=True) is False
    assert (cfg.packing_strategy, cfg.padding_free) == ("bfd", None)


def test_unsloth_kernels_do_not_count_for_a_transformers_loaded_model(dispatch):
    dispatch.HAS_XFORMERS = True
    assert _varlen_attention_available(_model(), unsloth_loaded=False) is False
    cfg = _cfg()
    assert _keep_batches_rectangular(cfg, _model(), unsloth_loaded=False) is True


@pytest.mark.parametrize("attn", ["flash_attention_2", "flash_attention_3"])
def test_flash_attention_models_are_left_alone(dispatch, attn):
    cfg = _cfg()
    assert _keep_batches_rectangular(cfg, _model(attn), unsloth_loaded=False) is False
    assert cfg.packing_strategy == "bfd"


def test_missing_config_fields_do_not_raise(dispatch):
    cfg = SimpleNamespace(packing=True)  # an SFTConfig without the newer fields
    assert _keep_batches_rectangular(cfg, SimpleNamespace(), unsloth_loaded=True) is True
    assert cfg._unsloth_disable_auto_packing is True
