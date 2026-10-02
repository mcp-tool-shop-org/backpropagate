# Unsloth's SDPA path uses PyTorch's memory-efficient attention kernel.
"""``_prefer_efficient_sdpa`` (trainer.py).

Without flash-attention or xFormers, Unsloth calls PyTorch's
``scaled_dot_product_attention`` with ``enable_gqa=True`` for grouped-query
models. On builds without the flash kernel (every Windows wheel) PyTorch then
falls back to its "math" kernel and materialises ``heads x seq x seq`` scores
(measured: 2 GiB for one 2,048-token row at 32 heads, against 0.03 GiB with
the key/value heads expanded). Unsloth's modules read a ``SDPA_HAS_GQA`` flag
at call time; clearing it selects the expansion path. These pin when the flag
is cleared and when it is left alone.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from backpropagate.trainer import _prefer_efficient_sdpa


@pytest.fixture
def unsloth(monkeypatch):
    """Stand-ins for the Unsloth modules that carry the flag."""
    mods = {
        "unsloth": SimpleNamespace(),
        "unsloth.utils.attention_dispatch": SimpleNamespace(
            SDPA_HAS_GQA=True, HAS_FLASH_ATTENTION=False, HAS_XFORMERS=False
        ),
        "unsloth.models.llama": SimpleNamespace(SDPA_HAS_GQA=True),
        "unsloth.models.qwen3": SimpleNamespace(SDPA_HAS_GQA=True),
        "unsloth.models.loader": SimpleNamespace(),  # no flag: untouched
    }
    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return mods


def test_clears_the_flag_in_every_unsloth_module(unsloth):
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is True
    assert unsloth["unsloth.utils.attention_dispatch"].SDPA_HAS_GQA is False
    assert unsloth["unsloth.models.llama"].SDPA_HAS_GQA is False
    assert unsloth["unsloth.models.qwen3"].SDPA_HAS_GQA is False
    assert not hasattr(unsloth["unsloth.models.loader"], "SDPA_HAS_GQA")


def test_second_call_changes_nothing(unsloth):
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is True
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is False


def test_transformers_loaded_model_is_left_alone(unsloth):
    assert _prefer_efficient_sdpa(unsloth_loaded=False) is False
    assert unsloth["unsloth.models.llama"].SDPA_HAS_GQA is True


@pytest.mark.parametrize("flag", ["HAS_FLASH_ATTENTION", "HAS_XFORMERS"])
def test_a_varlen_kernel_is_left_alone(unsloth, flag):
    setattr(unsloth["unsloth.utils.attention_dispatch"], flag, True)
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is False
    assert unsloth["unsloth.models.llama"].SDPA_HAS_GQA is True


def test_unsloth_without_the_flag_is_a_no_op(unsloth, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "unsloth.utils.attention_dispatch", SimpleNamespace(HAS_XFORMERS=False)
    )
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is False
    assert unsloth["unsloth.models.llama"].SDPA_HAS_GQA is True


def test_unsloth_not_imported_is_a_no_op(monkeypatch):
    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name)
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is False


def test_other_packages_and_torch_versions_already_without_gqa(unsloth, monkeypatch):
    # A module outside Unsloth with the same attribute is not ours to change,
    # and an Unsloth that already reports no GQA support needs no change.
    other = SimpleNamespace(SDPA_HAS_GQA=True)
    monkeypatch.setitem(sys.modules, "unslothish.models", other)
    for mod in unsloth.values():
        if hasattr(mod, "SDPA_HAS_GQA"):
            mod.SDPA_HAS_GQA = False
    assert _prefer_efficient_sdpa(unsloth_loaded=True) is False
    assert other.SDPA_HAS_GQA is True
