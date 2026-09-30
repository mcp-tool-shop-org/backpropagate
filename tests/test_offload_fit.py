"""The measured fit check that gates ``full_ft_offload`` (CPU-only unit tests).

The constants live in ``backpropagate/offload_engine.py``. The receipts they
cite are versioned in ``docs/receipts/2026-09-30-offload/``, and
``TestModelCoversEveryReceipt`` re-reads those receipts: if a constant is
edited so that it no longer covers a measured run, these tests fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import backpropagate.offload_engine as oe
import backpropagate.trainer as t
from backpropagate.exceptions import FullFinetuneModelTooLargeError, OffloadDoesNotFitError

RECEIPTS = Path(__file__).resolve().parent.parent / "docs" / "receipts" / "2026-09-30-offload"

# Architecture of the measured models, taken from their HF config.json. The
# fields are: params, vocab, hidden, layers, tied embeddings. They feed the
# model's structural term (embedding/head + one decoder layer, in bf16).
ARCH = {
    "Qwen/Qwen2.5-1.5B-Instruct": (1_543_714_304, 151_936, 1536, 28, True),
    "HuggingFaceTB/SmolLM3-3B": (3_075_098_624, 128_256, 2048, 36, True),
    "Qwen/Qwen3-4B-Instruct-2507": (4_022_468_096, 151_936, 2560, 36, True),
    "Qwen/Qwen2.5-7B-Instruct": (7_615_616_512, 152_064, 3584, 28, False),
}


def shape(model_id: str) -> tuple[int, int, int]:
    params, vocab, hidden, layers, tied = ARCH[model_id]
    emb = vocab * hidden
    root = emb * 2 * (1 if tied else 2)
    layer = (params - emb * (1 if tied else 2)) // layers * 2
    return params, root, layer


def fit(model_id: str, *, avail: float, vram: float = 31.36, tokens: int = 512) -> dict:
    params, root, layer = shape(model_id)
    return oe.check_offload_fit(
        params=params, root_unit_bytes=root, layer_bytes=layer, tokens=tokens,
        host_total_gib=avail, host_available_gib=avail, vram_total_gib=vram,
    )


class TestFitCheck:
    def test_7b_fits_a_64gb_host_with_a_32gb_card(self):
        r = fit("Qwen/Qwen2.5-7B-Instruct", avail=60.0)
        assert r["fits"], r
        assert r["host_ram_required_gib"] == pytest.approx(38.5, abs=0.1)
        assert r["vram_required_gib"] == pytest.approx(6.4, abs=0.1)

    def test_7b_does_not_fit_a_28gib_wsl2_cap(self):
        r = fit("Qwen/Qwen2.5-7B-Instruct", avail=28.0)
        assert not r["fits_host_ram"] and r["fits_vram"]
        msg = str(OffloadDoesNotFitError("Qwen/Qwen2.5-7B-Instruct", r))
        assert "38.5 GiB" in msg and "28.0 GiB available" in msg
        assert ".wslconfig" in msg and "wsl --shutdown" in msg
        assert "mode='lora'" in msg and "smaller model" in msg
        assert "VRAM" not in msg  # only the failing side is reported

    def test_4b_fits_a_28gib_wsl2_cap(self):
        r = fit("Qwen/Qwen3-4B-Instruct-2507", avail=28.0)
        assert r["fits"], r

    def test_13b_needs_about_59gib(self):
        """13B needs ~58.6 GiB of host RAM (an extrapolation past the largest
        measured run, 7.6B). A 64 GiB machine with the usual ~56 GiB available
        after the OS fails; an idle one with >= 59 GiB passes."""
        need = oe.offload_host_ram_required_gib(13e9)
        assert need == pytest.approx(58.6, abs=0.1)
        busy = oe.check_offload_fit(params=13e9, root_unit_bytes=None, layer_bytes=None,
                                    tokens=512, host_total_gib=62.7, host_available_gib=56.0,
                                    vram_total_gib=31.36)
        assert not busy["fits_host_ram"]
        idle = oe.check_offload_fit(params=13e9, root_unit_bytes=None, layer_bytes=None,
                                    tokens=512, host_total_gib=62.7, host_available_gib=59.0,
                                    vram_total_gib=31.36)
        assert idle["fits_host_ram"]

    def test_vram_side_fails_on_a_small_card_at_long_context(self):
        r = fit("Qwen/Qwen2.5-7B-Instruct", avail=60.0, vram=8.0, tokens=4096)
        assert r["fits_host_ram"] and not r["fits_vram"]
        msg = str(OffloadDoesNotFitError("Qwen/Qwen2.5-7B-Instruct", r))
        assert "4096 tokens/step" in msg and "lower max_seq_length" in msg

    def test_unknown_sides_are_not_judged(self):
        r = oe.check_offload_fit(params=7.6e9, root_unit_bytes=None, layer_bytes=None,
                                 tokens=512, host_total_gib=None, host_available_gib=None,
                                 vram_total_gib=None)
        assert r["fits"] and r["vram_required_gib"] is None

    def test_error_keeps_the_catalogued_code_and_parent(self):
        err = OffloadDoesNotFitError("m", fit("Qwen/Qwen2.5-7B-Instruct", avail=20.0))
        assert err.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE"
        assert isinstance(err, FullFinetuneModelTooLargeError)
        assert err.report["host_ram_available_gib"] == 20.0


class TestModelCoversEveryReceipt:
    """The model must be >= every measured peak, and must admit every run that
    actually trained under a VRAM cap."""

    @pytest.mark.parametrize("name", [
        "q15b_offload_reg", "smollm3_offload_reg", "qwen3_4b_offload_reg", "q7b_final_receipt",
    ])
    def test_host_ram_covers_measured_peak(self, name):
        r = json.loads((RECEIPTS / f"{name}.json").read_text(encoding="utf-8"))
        peak = r.get("peak_rss_total_gb") or r["peak_rss_train_gb"]
        assert oe.offload_host_ram_required_gib(r["params"]) >= peak

    @pytest.mark.parametrize("name", [
        "q15b_offload_reg", "q15b_sr_retention", "smollm3_offload_reg", "qwen3_4b_offload_reg",
        "q7b_final_receipt", "q7b_seq2048",
    ])
    def test_vram_covers_measured_peak(self, name):
        r = json.loads((RECEIPTS / f"{name}.json").read_text(encoding="utf-8"))
        _, root, layer = shape(r["model"])
        assert oe.offload_vram_required_gib(root, layer, r["seq"]) >= r["peak_vram_alloc_gb"]

    def test_vram_covers_the_batch4_quality_run(self):
        rows = [json.loads(line) for line in (RECEIPTS / "quality.jsonl").read_text().splitlines()]
        run = next(r for r in rows if r.get("engine") == "offload")
        _, root, layer = shape(run["model"])
        tokens = run["batch"] * run["seq"]
        assert oe.offload_vram_required_gib(root, layer, tokens) >= run["peak_vram_alloc_gb"]

    @pytest.mark.parametrize("name", ["q7b_cap8", "smollm3_cap6", "smollm3_cap8", "qwen3_4b_cap8"])
    def test_capped_runs_that_trained_are_admitted(self, name):
        r = json.loads((RECEIPTS / f"{name}.json").read_text(encoding="utf-8"))
        assert r["losses"], "the capped run must have trained"
        _, root, layer = shape(r["model"])
        assert oe.offload_vram_required_gib(root, layer, r["seq"]) <= r["vram_cap_gb"]


class TestTrainerGate:
    @pytest.fixture
    def host(self, monkeypatch):
        def set_ram(avail: float, vram: float = 31.36) -> None:
            monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: (avail, avail))
            monkeypatch.setattr(t, "_detect_total_vram_gb", lambda: vram)
        return set_ram

    def test_construction_fails_fast_on_ram(self, host, tmp_path):
        host(28.0)
        with pytest.raises(OffloadDoesNotFitError) as ei:
            t.Trainer(model="Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_offload=True,
                      use_unsloth=False, output_dir=str(tmp_path), report_to="none")
        assert ".wslconfig" in str(ei.value)

    def test_construction_passes_with_enough_ram(self, host, tmp_path):
        host(60.0)
        t.Trainer(model="Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_offload=True,
                  use_unsloth=False, output_dir=str(tmp_path), report_to="none")

    def test_explicit_override_is_the_escape_hatch(self, host, tmp_path, caplog):
        host(28.0)
        trainer = t.Trainer(model="Qwen/Qwen2.5-7B-Instruct", mode="full", full_ft_offload=True,
                            full_ft_ceiling_billions=8.0, use_unsloth=False,
                            output_dir=str(tmp_path), report_to="none")
        assert trainer.full_ft_ceiling_billions == 8.0
        assert any("overrides it" in rec.getMessage() for rec in caplog.records)

    def test_train_time_check_uses_the_real_model_shape(self, host, tmp_path):
        """Loaded model -> exact params / embedding / layer sizes, checked before
        any weight moves. A tiny model on a 4 GiB 'host' still fails, because the
        fixed term (10.1 GiB) does not fit; that proves the path runs."""
        from transformers import LlamaConfig, LlamaForCausalLM

        host(60.0)
        trainer = t.Trainer(model="some-org/unknown-model", mode="full", full_ft_offload=True,
                            use_unsloth=False, output_dir=str(tmp_path), report_to="none")
        cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        trainer._model = LlamaForCausalLM(cfg)
        trainer._is_loaded = True
        report = trainer._enforce_offload_fit_for_model()
        assert report["params_billions"] == pytest.approx(
            sum(p.numel() for p in trainer._model.parameters()) / 1e9, abs=1e-3
        )
        host(4.0)
        with pytest.raises(OffloadDoesNotFitError):
            trainer._enforce_offload_fit_for_model()

    def test_pure_gpu_path_never_runs_the_offload_check(self, host, tmp_path, monkeypatch):
        host(4.0)  # would fail any offload check
        monkeypatch.setattr(oe, "check_offload_fit", lambda **k: pytest.fail("offload check ran"))
        t.Trainer(model="HuggingFaceTB/SmolLM3-3B", mode="full", full_ft_offload=False,
                  use_unsloth=False, output_dir=str(tmp_path), report_to="none")
        assert t._full_ft_ceiling_for_vram(32) == 6.0  # the pure-GPU table is unchanged


class TestEstimateVramOffload:
    def test_host_ram_uses_the_measured_model(self):
        est = t.estimate_vram("Qwen/Qwen2.5-7B-Instruct", mode="full", offload=True,
                              param_count_billions=7.6156, hidden_dim=3584, num_layers=28,
                              max_seq_length=512)
        assert est.host_ram_gb == pytest.approx(38.5, abs=0.1)
        # GPU side with vocab 152064 (untied): matches the fit check (6.4 GiB).
        assert est.total_gb == pytest.approx(6.4, abs=0.15)
