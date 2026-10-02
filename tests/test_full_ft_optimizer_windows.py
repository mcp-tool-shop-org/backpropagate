# Windows: full fine-tuning avoids the paged 8-bit optimizer when it can.
"""``_full_ft_needs_paged_optimizer`` and the ``full_ft_paged_optim`` switch.

``paged_adamw_8bit`` keeps its state in CUDA managed memory. On Windows the
display driver pages that memory, and a Llama 3.2 1B full fine-tune using it
froze the desktop twice on an RTX 5090 (2026-10-02) at 12.7 GB of 32 GB. The
same run with the non-paged ``adamw_8bit`` used 10.1 GB and stayed smooth.

So on Windows the trainer picks the non-paged optimizer whenever the
gradients and optimizer state fit in free GPU memory, and keeps the paged one
(with a warning) only when they do not.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backpropagate import trainer as trainer_mod
from backpropagate.trainer import _full_ft_needs_paged_optimizer

GIB = 1024**3


def _model(trainable: float, vocab: int = 128256, hidden: int = 2048, frozen: float = 0.0):
    params = [SimpleNamespace(numel=lambda n=trainable: int(n), requires_grad=True)]
    if frozen:
        params.append(SimpleNamespace(numel=lambda n=frozen: int(n), requires_grad=False))
    return SimpleNamespace(
        parameters=lambda: iter(params),
        config=SimpleNamespace(vocab_size=vocab, hidden_size=hidden),
    )


@pytest.fixture
def windows_gpu(monkeypatch):
    """Windows with a CUDA card; ``free`` (GiB) is settable."""
    state = {"free": 28.0}
    monkeypatch.setattr(trainer_mod, "_is_windows", lambda: True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.mem_get_info", lambda *a: (int(state["free"] * GIB), 32 * GIB)
    )
    return state


def test_llama_1b_fits_without_paging_on_a_32gb_card(windows_gpu):
    # Measured: 9.56 GiB peak with adamw_8bit, 2.31 GiB of it the weights.
    assert _full_ft_needs_paged_optimizer(_model(1.236e9), 1, 512) is False


def test_a_3b_model_needs_paging_on_a_16gb_card(windows_gpu):
    windows_gpu["free"] = 8.5  # 16 GB card, weights (5.7 GiB) loaded, desktop running
    assert _full_ft_needs_paged_optimizer(_model(3.075e9), 4, 512) is True


def test_a_3b_model_fits_without_paging_on_a_32gb_card(windows_gpu):
    windows_gpu["free"] = 24.5
    assert _full_ft_needs_paged_optimizer(_model(3.075e9), 4, 512) is False


def test_the_batch_counts(windows_gpu):
    windows_gpu["free"] = 12.0
    model = _model(1.236e9)
    assert _full_ft_needs_paged_optimizer(model, 1, 512) is False
    # 8 rows of 2,048 tokens add about 8.8 GiB of logits and activations.
    assert _full_ft_needs_paged_optimizer(model, 8, 2048) is True


def test_only_trainable_parameters_are_sized(windows_gpu):
    windows_gpu["free"] = 4.0
    assert _full_ft_needs_paged_optimizer(_model(0.1e9, frozen=7e9), 1, 512) is False


def test_not_windows_keeps_the_default(monkeypatch):
    monkeypatch.setattr(trainer_mod, "_is_windows", lambda: False)
    assert _full_ft_needs_paged_optimizer(_model(1e9), 1, 512) is None


def test_no_cuda_keeps_the_default(monkeypatch):
    monkeypatch.setattr(trainer_mod, "_is_windows", lambda: True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert _full_ft_needs_paged_optimizer(_model(1e9), 1, 512) is None


def test_a_model_that_cannot_be_sized_keeps_the_default(windows_gpu):
    broken = SimpleNamespace(parameters=lambda: (_ for _ in ()).throw(RuntimeError("no")))
    assert _full_ft_needs_paged_optimizer(broken, 1, 512) is None
    assert _full_ft_needs_paged_optimizer(_model(0), 1, 512) is None


def test_a_model_without_a_config_still_gets_an_answer(windows_gpu):
    bare = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(numel=lambda: int(1e9), requires_grad=True)])
    )
    assert _full_ft_needs_paged_optimizer(bare, 1, 512) is False


# ---- the switch in _build_sft_config ---------------------------------------------------


def _build(caplog=None, **kw):
    captured: dict = {}

    class _StubSFTConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    args = {
        "output_dir": "./out", "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1, "max_steps": 10, "learning_rate": 2e-5,
        "warmup_steps": 2, "max_seq_length": 512, "seed": 42,
        "lr_scheduler_type": "cosine", "logging_steps": 10, "mode": "full",
        "optim": "adamw_8bit",
    }
    args.update(kw)
    with patch.dict("sys.modules", {"trl": MagicMock(SFTConfig=_StubSFTConfig)}), \
            patch("torch.cuda.is_available", return_value=True):
        trainer_mod._build_sft_config(**args)
    return captured


def test_fits_uses_the_non_paged_optimizer_even_on_a_small_card(caplog):
    # A card under 24 GB: the detector would upgrade adamw_8bit to the paged
    # variant, which is exactly what must not happen here.
    props = SimpleNamespace(total_memory=16 * GIB)
    with patch("torch.cuda.get_device_properties", return_value=props), \
            caplog.at_level(logging.INFO, logger=trainer_mod.logger.name):
        captured = _build(full_ft_paged_optim=False)
    assert captured["optim"] == "adamw_8bit"
    assert any("adamw_8bit" in r.getMessage() for r in caplog.records)


def test_does_not_fit_keeps_the_paged_optimizer_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=trainer_mod.logger.name):
        captured = _build(full_ft_paged_optim=True)
    assert captured["optim"] == "paged_adamw_8bit"
    assert any(
        r.levelno == logging.WARNING and "stop responding" in r.getMessage()
        for r in caplog.records
    )


def test_undecided_keeps_the_paged_optimizer_without_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=trainer_mod.logger.name):
        captured = _build()
    assert captured["optim"] == "paged_adamw_8bit"
    assert not any("stop responding" in r.getMessage() for r in caplog.records)


def test_an_explicit_optimizer_is_never_overridden():
    assert _build(optim="adamw_torch", full_ft_paged_optim=False)["optim"] == "adamw_torch"
    assert _build(optim="paged_adamw_8bit", full_ft_paged_optim=False)["optim"] == (
        "paged_adamw_8bit"
    )


def test_lora_mode_ignores_the_switch():
    props = SimpleNamespace(total_memory=32 * GIB)
    with patch("torch.cuda.get_device_properties", return_value=props):
        assert _build(mode="lora", full_ft_paged_optim=False)["optim"] == "adamw_8bit"
        assert _build(mode="lora", full_ft_paged_optim=True)["optim"] == "adamw_8bit"
