"""Regression tests for the PEFT adapter extraction path (real tiny PEFT model).

Nothing is mocked: a real ``peft.get_peft_model`` adapter over a real tiny Llama.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate.multi_run import MultiRunTrainer
from tests.test_multi_run_cov_support import N_A, N_B, FakeInnerTrainer, build_peft_llama


class TestLoraExtractionOnRealPeftModel:
    """Regression for the bug fixed alongside these tests: with transformers>=5 a
    ``peft.get_peft_model`` model *has* a ``get_adapter_state_dict`` attribute (the
    transformers PEFT mixin) that raises ``ValueError("No adapter loaded")``, so
    ``hasattr`` alone cannot select the extraction path."""

    def test_extraction_falls_back_to_manual_scan_on_a_real_peft_model(self):
        model = build_peft_llama()
        mrt = MultiRunTrainer(model="tiny")
        mrt._trainer = FakeInnerTrainer(model)

        state = mrt._get_lora_state_dict()

        expected = {n for n, _ in model.named_parameters() if "lora_" in n}
        assert set(state) == expected
        assert len(state) == N_A + N_B
        for name, tensor in state.items():
            assert torch.equal(tensor, dict(model.named_parameters())[name])

    def test_startup_invariant_check_passes_on_a_real_peft_model(self):
        mrt = MultiRunTrainer(model="tiny")
        mrt._trainer = FakeInnerTrainer(build_peft_llama())

        mrt._verify_peft_api()  # must not raise PEFT_API_INCOMPATIBLE

    def test_extraction_prefers_the_peft_api_when_it_works(self):
        class _WithApi:
            def get_adapter_state_dict(self):
                return {"x.lora_A.w": torch.ones(1), "x.lora_B.w": torch.zeros(1)}

        mrt = MultiRunTrainer(model="tiny")
        mrt._trainer = type("T", (), {"_model": _WithApi()})()

        state = mrt._get_lora_state_dict()

        assert set(state) == {"x.lora_A.w", "x.lora_B.w"}
