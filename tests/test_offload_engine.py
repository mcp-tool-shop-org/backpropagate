"""CPU unit pins for the direct-FSDP2 offload engine (backpropagate.offload_engine).

The real-GPU proof lives in tests/test_full_ft_offload_smoke.py and in
scripts/pod_offload_7b.sh. These pins keep the optimizer math, the bf16
stochastic-rounding contract, and the Trainer routing honest in CI, where no
GPU or NCCL is available.
"""

from __future__ import annotations

import pytest
import torch

import backpropagate.offload_engine as oe


def _opt(params, **kw):
    kw.setdefault("device", "cpu")
    return oe.OffloadAdafactor(params, lr=kw.pop("lr", 1e-3), **kw)


class TestStochasticRounding:
    def test_unbiased_for_sub_ulp_updates(self):
        """A +1e-4 update to 1.0 is ~1/78 of a bf16 ulp (2^-7). Nearest rounding
        drops it every time; stochastic rounding keeps it in expectation."""
        torch.manual_seed(0)
        target = torch.full((200_000,), 1.0 + 1e-4, dtype=torch.float32)
        nearest = target.to(torch.bfloat16).float()
        assert torch.all(nearest == 1.0)
        out = torch.empty(target.shape, dtype=torch.bfloat16)
        oe.stochastic_round_to_bf16_(out, target)
        assert out.float().mean().item() == pytest.approx(1.0 + 1e-4, abs=2e-5)
        assert (out.float() != 1.0).any()

    def test_exact_values_are_unchanged(self):
        exact = torch.tensor([1.0, -2.5, 0.0, 3.0], dtype=torch.float32)
        out = torch.empty(4, dtype=torch.bfloat16)
        oe.stochastic_round_to_bf16_(out, exact)
        assert torch.equal(out.float(), exact)


class TestOffloadAdafactor:
    def test_bf16_updates_survive_at_full_ft_lr(self):
        """The precision gate. At the full-FT LR (2e-5) a single update is a fraction
        of a bf16 ulp. Round-to-nearest DROPS most of it, a systematic bias that
        silently stops learning. Stochastic rounding changes only some elements per
        step, but the applied update matches the fp32 update in expectation."""
        torch.manual_seed(0)
        w0 = (torch.randn(512, 512) * 0.02).to(torch.bfloat16)
        grads = torch.randn(512, 512).to(torch.bfloat16)

        def applied(dtype, sr):
            p = torch.nn.Parameter(w0.clone().to(dtype))
            p.grad = grads.clone().to(dtype)
            _opt([p], lr=2e-5, stochastic_rounding=sr).step()
            return p.data.float() - w0.float()

        intended = applied(torch.float32, sr=False)  # fp32 reference update
        direction = intended.sign()
        target = intended.abs().sum().item()
        kept_nearest = (applied(torch.bfloat16, sr=False) * direction).sum().item() / target
        kept_sr = (applied(torch.bfloat16, sr=True) * direction).sum().item() / target
        assert kept_sr == pytest.approx(1.0, abs=0.05), kept_sr
        assert kept_nearest < 0.7, kept_nearest

    def test_update_direction_descends(self):
        p = torch.nn.Parameter(torch.zeros(8, 4))
        p.grad = torch.ones(8, 4)
        _opt([p], lr=0.1).step()
        assert torch.all(p.data < 0)

    def test_chunked_step_matches_unchunked(self, monkeypatch):
        torch.manual_seed(1)
        w = torch.randn(64, 32)
        g = torch.randn(64, 32)
        results = []
        for chunk in (1 << 26, 100):  # one chunk vs. ~21 chunks of 3 rows
            monkeypatch.setattr(oe, "_MAX_CHUNK_NUMEL", chunk)
            p = torch.nn.Parameter(w.clone())
            p.grad = g.clone()
            opt = _opt([p], lr=1e-2)
            opt.step()
            p.grad = g.clone() * 0.5
            opt.step()
            results.append(p.data.clone())
        torch.testing.assert_close(results[0], results[1])

    def test_state_is_factored(self):
        p = torch.nn.Parameter(torch.randn(128, 64))
        p.grad = torch.randn(128, 64)
        opt = _opt([p])
        opt.step()
        st = opt.state[p]
        assert st["row"].shape == (128,) and st["col"].shape == (64,)
        assert "v" not in st

    def test_vector_params_use_full_second_moment(self):
        p = torch.nn.Parameter(torch.ones(16))
        p.grad = torch.ones(16)
        opt = _opt([p], lr=1e-2)
        opt.step()
        assert opt.state[p]["v"].shape == (16,)
        assert torch.all(p.data < 1.0)

    def test_update_is_clipped(self):
        """Update clipping bounds the per-element step to ~lr (RMS 1)."""
        p = torch.nn.Parameter(torch.zeros(32, 32))
        p.grad = torch.randn(32, 32) * 1e6
        _opt([p], lr=1e-3).step()
        rms = p.data.pow(2).mean().sqrt().item()
        assert rms <= 1e-3 * 1.01

    def test_rejects_nonpositive_lr(self):
        with pytest.raises(ValueError):
            oe.OffloadAdafactor([torch.nn.Parameter(torch.zeros(2))], lr=0.0, device="cpu")


class TestSchedulesAndLayers:
    def test_lr_factor_warmup_then_cosine(self):
        assert oe._lr_factor(0, 10, 2, "cosine") == pytest.approx(0.5)
        assert oe._lr_factor(1, 10, 2, "cosine") == pytest.approx(1.0)
        assert oe._lr_factor(2, 10, 2, "cosine") == pytest.approx(1.0)
        assert oe._lr_factor(9, 10, 2, "cosine") < 0.1
        assert oe._lr_factor(5, 10, 0, "constant") == 1.0

    def test_decoder_layers_found_on_a_tiny_hf_model(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4)
        model = LlamaForCausalLM(cfg)
        layers = oe._decoder_layers(model)
        assert len(layers) == 3
        assert type(layers[0]).__name__ == "LlamaDecoderLayer"


def test_trainer_routes_offload_to_the_direct_engine(monkeypatch, tmp_path):
    """mode='full' + full_ft_offload=True must bypass SFTTrainer entirely."""
    import json

    import backpropagate.trainer as t

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(t, "_ensure_fsdp_runtime", lambda: None)
    trainer = t.Trainer(model="HuggingFaceTB/SmolLM2-135M-Instruct", use_unsloth=False,
                        mode="full", full_ft_offload=True, output_dir=str(tmp_path / "o"),
                        report_to="none")
    monkeypatch.setattr(trainer, "load_model", lambda: setattr(trainer, "_is_loaded", True))
    monkeypatch.setattr(trainer, "_load_dataset", lambda *a, **k: ["row"])
    calls = {}

    def fake_run(model, tok, ds, **kw):
        calls.update(kw)
        return {"model": model, "losses": [2.0, 1.5], "step_times": [0.1, 0.1],
                "samples_seen": 2, "duration_seconds": 0.2, "optimizer": None}

    monkeypatch.setattr(oe, "run_offload_training", fake_run)
    monkeypatch.setattr(t.Trainer, "_build_trainer", lambda *a, **k: pytest.fail("SFTTrainer built"))
    run = trainer.train(str(data), steps=2)
    assert run.final_loss == 1.5
    assert run.metadata["engine"] == "fsdp2-direct"
    assert calls["steps"] == 2
