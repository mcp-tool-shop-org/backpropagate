"""Non-mocked round-trip coverage for the SLAO eval-gated merge (v1.6 hardening).

This is the load-bearing test for the eval-gate-core wave: it exercises the
REAL adapter write -> ``PeftModel.from_pretrained`` load path and the REAL
last-10% held-out derivation WITHOUT mocking ``evaluate_run`` — the exact seam
the shipped-but-non-functional eval gate failed on (TRAINER-A-001 wrote adapter
weights with NO ``adapter_config.json`` so PEFT could never load them;
TRAINER-A-002 never materialized the advertised reserved-holdout default).

Two tiers:

* **Network-free** — assert ``MultiRunTrainer._write_eval_adapter_config`` writes
  a well-formed ``adapter_config.json`` and that ``_derive_last_decile_heldout``
  produces a non-empty held-out text set with the documented degenerate guard.
  These never touch the network.
* **Model-dependent** (``@pytest.mark.skipif`` when the tiny model can't load) —
  drive a REAL tiny ``PeftModel`` through ``_evaluate_accumulator`` with
  ``eval_heldout_path=None`` and an un-mocked ``evaluate_run``, asserting the
  written dir loads via ``PeftModel.from_pretrained`` and a finite held-out loss
  comes back.
"""

from __future__ import annotations

import json

import pytest

# A tiny CPU-runnable causal LM. Downloaded once if the network is available;
# the model-dependent tests skip cleanly when it is not.
_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _peft_available() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _build_tiny_peft_model():
    """Load the tiny base model + attach a small LoRA adapter (real PEFT).

    Returns ``(peft_model, lora_config)`` or raises so the caller can skip.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, cfg)
    return model, cfg


# Resolve model availability once at import time so skipif messages are clean.
try:
    _TINY_MODEL_OK = False
    if _peft_available():
        _m, _c = _build_tiny_peft_model()
        del _m, _c
        _TINY_MODEL_OK = True
    _SKIP_REASON = "" if _TINY_MODEL_OK else f"tiny model {_TINY_MODEL} unavailable"
except Exception as _exc:  # offline / no cache / download blocked
    _TINY_MODEL_OK = False
    _SKIP_REASON = f"tiny model {_TINY_MODEL} unavailable ({type(_exc).__name__})"

requires_tiny_model = pytest.mark.skipif(
    not _TINY_MODEL_OK, reason=_SKIP_REASON or "tiny model unavailable"
)
requires_peft = pytest.mark.skipif(
    not _peft_available(), reason="peft/torch/transformers not importable"
)


def _make_trainer(tmp_path, **config_overrides):
    """A MultiRunTrainer wired with a real CheckpointManager + a real
    SLAOMerger, ready to call the eval-gate helpers directly."""
    from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy
    from backpropagate.multi_run import MergeMode, MultiRunConfig, MultiRunTrainer

    cfg = MultiRunConfig(
        merge_mode=MergeMode.SLAO,
        checkpoint_dir=str(tmp_path),
        **config_overrides,
    )
    trainer = MultiRunTrainer(model=_TINY_MODEL, config=cfg)
    trainer._run_id = "rt-run"
    trainer._checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(tmp_path), policy=CheckpointPolicy()
    )
    return trainer


class _FakeHFDataset:
    """Minimal HF-Dataset stand-in: ``len`` + ``select`` over ChatML rows."""

    def __init__(self, n=20):
        self._rows = [{"text": f"<|im_start|>user\nq{i}<|im_end|>"} for i in range(n)]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]

    def select(self, indices):
        sub = _FakeHFDataset(0)
        sub._rows = [self._rows[i] for i in indices]
        return sub

    def __iter__(self):
        return iter(self._rows)


# =============================================================================
# Network-free: held-out derivation (TRAINER-A-002)
# =============================================================================

class TestLastDecileHeldout:
    def test_derives_non_empty_last_decile(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        texts = trainer._derive_last_decile_heldout(_FakeHFDataset(n=20))
        # 20 rows -> last 10% == 2 rows, both non-empty.
        assert len(texts) == 2
        assert texts == ["<|im_start|>user\nq18<|im_end|>",
                         "<|im_start|>user\nq19<|im_end|>"]

    def test_degenerate_small_dataset_yields_at_least_one(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        # 3 rows -> 3 // 10 == 0 -> guard forces at least 1 (the final row).
        texts = trainer._derive_last_decile_heldout(_FakeHFDataset(n=3))
        assert len(texts) == 1
        assert texts == ["<|im_start|>user\nq2<|im_end|>"]

    def test_none_dataset_raises_unresolved(self, tmp_path):
        from backpropagate.exceptions import BackpropagateError

        trainer = _make_trainer(tmp_path)
        with pytest.raises(BackpropagateError) as exc:
            trainer._derive_last_decile_heldout(None)
        assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"

    def test_empty_dataset_raises_unresolved(self, tmp_path):
        from backpropagate.exceptions import BackpropagateError

        trainer = _make_trainer(tmp_path)
        with pytest.raises(BackpropagateError) as exc:
            trainer._derive_last_decile_heldout(_FakeHFDataset(n=0))
        assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"


# =============================================================================
# Network-free: adapter_config.json is written + well-formed (TRAINER-A-001)
# =============================================================================

class TestAdapterConfigWritten:
    @requires_peft
    def test_fallback_writes_well_formed_config(self, tmp_path):
        """Even with NO live PeftModel (a mocked inner trainer), the fallback
        path writes a valid, loadable-shape ``adapter_config.json``."""
        from unittest.mock import MagicMock

        trainer = _make_trainer(tmp_path)
        # A mocked inner trainer whose _model has no dict peft_config -> fallback.
        inner = MagicMock()
        inner.lora_r = 8
        inner.lora_alpha = 16
        # MagicMock attribute returns a MagicMock (not a dict) -> fallback path.
        trainer._trainer = inner

        adapter_dir = tmp_path / "lora"
        adapter_dir.mkdir()
        trainer._write_eval_adapter_config(adapter_dir)

        cfg_path = adapter_dir / "adapter_config.json"
        assert cfg_path.exists(), "adapter_config.json must be written"
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        # PEFT's adapter_config.json carries these load-bearing keys.
        assert data.get("peft_type") == "LORA"
        assert data.get("r") == 8
        assert data.get("lora_alpha") == 16
        assert data.get("base_model_name_or_path") == _TINY_MODEL
        # And PEFT can parse it back into a LoraConfig (well-formed contract).
        from peft import LoraConfig

        reloaded = LoraConfig.from_pretrained(str(adapter_dir))
        assert reloaded.r == 8

    @requires_tiny_model
    def test_live_peft_model_config_written(self, tmp_path):
        """With a REAL live PeftModel, the active LoraConfig is the source of
        truth and is persisted via save_pretrained."""
        from unittest.mock import MagicMock

        model, _cfg = _build_tiny_peft_model()
        trainer = _make_trainer(tmp_path)
        inner = MagicMock()
        inner._model = model  # real PeftModel -> dict peft_config branch
        trainer._trainer = inner

        adapter_dir = tmp_path / "lora"
        adapter_dir.mkdir()
        trainer._write_eval_adapter_config(adapter_dir)

        data = json.loads(
            (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
        assert data["peft_type"] == "LORA"
        assert data["r"] == 8
        assert data["lora_alpha"] == 16
        # target_modules from the live config survive (q_proj / v_proj).
        tm = data.get("target_modules") or []
        assert "q_proj" in tm and "v_proj" in tm


# =============================================================================
# Model-dependent: REAL write -> load round-trip + finite loss (the core)
# =============================================================================

@requires_tiny_model
class TestRealAdapterRoundTrip:
    def _real_accumulator_state(self, model):
        """Extract a real PEFT-format adapter state dict from the live model."""
        from peft.utils import get_peft_model_state_dict

        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

    def test_written_adapter_loads_via_peft_from_pretrained(self, tmp_path):
        """TRAINER-A-001 acceptance bar: the dir _write_eval_adapter_config +
        the saved weights produce loads via a REAL PeftModel.from_pretrained."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model, _cfg = _build_tiny_peft_model()
        trainer = _make_trainer(tmp_path)
        from unittest.mock import MagicMock

        inner = MagicMock()
        inner._model = model
        trainer._trainer = inner

        adapter_dir = tmp_path / "lora"
        adapter_dir.mkdir()
        torch.save(self._real_accumulator_state(model), adapter_dir / "adapter_model.bin")
        trainer._write_eval_adapter_config(adapter_dir)

        # The acceptance bar: a fresh base + this dir loads via PEFT.
        base2 = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
        loaded = PeftModel.from_pretrained(base2, str(adapter_dir))
        assert loaded is not None

    def test_evaluate_accumulator_real_eval_returns_finite_loss(self, tmp_path):
        """The full eval-gate inner loop with eval_heldout_path=None and an
        UN-MOCKED evaluate_run: derive last-10% holdout, write adapter + config,
        load via PEFT, compute a finite held-out loss."""
        import math

        model, _cfg = _build_tiny_peft_model()
        trainer = _make_trainer(tmp_path, eval_gate=True, eval_max_regression=0.0)
        from unittest.mock import MagicMock

        inner = MagicMock()
        inner._model = model
        inner.max_seq_length = 64
        trainer._trainer = inner

        accumulator = self._real_accumulator_state(model)
        full_dataset = _FakeHFDataset(n=20)

        # NOTE: evaluate_run is NOT patched — this exercises the real model load,
        # the real PeftModel.from_pretrained on the written adapter dir, and the
        # real held-out loss computation against the derived last-10% texts.
        result = trainer._evaluate_accumulator(
            accumulator, run_idx=2, full_dataset=full_dataset, phase="after"
        )

        assert result is not None
        assert result.held_out_loss is not None
        assert math.isfinite(result.held_out_loss)
        # Generations populate against the default prompt set.
        assert result.n_prompts > 0
