# VRAM calibration: measure on the running GPU, store, and reuse.
"""``backpropagate.vram_calibration`` and its wiring (CPU-only tests).

The probes themselves need a GPU and are covered by
``tests/test_vram_calibration_gpu.py`` (run by hand on the rig). These pin
everything around them: the row fit, the per-machine store, how
``estimate_vram`` uses a stored measurement, the CLI flags, and the parent's
handling of the child's output.
"""

from __future__ import annotations

import io
import json

import pytest

from backpropagate import vram_calibration as vc
from backpropagate.trainer import estimate_vram

GIB = vc.GIB
MACHINE = {"gpu": "Test GPU", "vram_gib": 24.0, "torch": "2.10", "transformers": "5.5",
           "unsloth": "2026.5", "varlen_attention": False}


def _cal(**kw) -> vc.Calibration:
    base = {
        "model": "org/m-1B", "mode": "lora", "base_4bit": True, "machine": MACHINE,
        "load_gib": 1.0, "floor_gib": 1.0, "quad_bytes": 500.0, "lin_bytes": 70_000.0,
        "probe_trainable_params": 1_000_000, "max_residual_pct": 1.0,
        "measured_at": "2026-10-02T10:00:00",
    }
    base.update(kw)
    return vc.Calibration(**base)


# ---- the fit ----------------------------------------------------------------------


def _probe(batch, seq, quad, lin):
    return {"batch": batch, "seq": seq,
            "overhead_gib": batch * (quad * seq * seq + lin * seq) / GIB}


def test_fit_recovers_both_terms_from_two_sequence_lengths():
    probes = [_probe(2, 2048, 512.0, 70_000.0), _probe(4, 2048, 512.0, 70_000.0),
              _probe(8, 1024, 512.0, 70_000.0)]
    quad, lin, resid = vc.fit_rows(probes, formula_quad=400.0, formula_lin=50_000.0)
    assert quad == pytest.approx(512.0, rel=1e-3)
    assert lin == pytest.approx(70_000.0, rel=1e-3)
    assert resid < 0.5


def test_fit_with_one_sequence_length_scales_the_formula_split():
    probes = [_probe(1, 2048, 600.0, 60_000.0), _probe(2, 2048, 600.0, 60_000.0)]
    quad, lin, _resid = vc.fit_rows(probes, formula_quad=400.0, formula_lin=40_000.0)
    assert quad / lin == pytest.approx(400.0 / 40_000.0)  # the formula's split
    assert 2 * (quad * 2048**2 + lin * 2048) / GIB == pytest.approx(probes[1]["overhead_gib"])


def test_fit_ignores_oom_probes_and_returns_none_without_data():
    assert vc.fit_rows([{"batch": 4, "seq": 2048, "oom": True}], 400.0, 40_000.0) is None
    assert vc.fit_rows([], 400.0, 40_000.0) is None


def test_rows_gib_is_none_until_rows_are_measured():
    assert _cal(quad_bytes=None, lin_bytes=None).rows_gib(4, 2048) is None
    assert _cal().rows_measured is True
    assert _cal().rows_gib(2, 1024) == pytest.approx(2 * (500 * 1024**2 + 70_000 * 1024) / GIB)


# ---- the store ----------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    monkeypatch.setenv("BACKPROPAGATE_VRAM_CALIBRATION", str(path))
    return path


def test_store_roundtrip_is_keyed_by_machine_model_mode_and_precision(store):
    vc.save(_cal())
    assert store.exists()
    assert vc.lookup("org/m-1B", "lora", True, machine=MACHINE).load_gib == 1.0
    assert vc.lookup("ORG/M-1B", "lora", True, machine=MACHINE) is not None  # case-insensitive id
    assert vc.lookup("org/m-1B", "lora", False, machine=MACHINE) is None  # 16-bit base
    assert vc.lookup("org/m-1B", "full", True, machine=MACHINE) is None
    assert vc.lookup("org/other", "lora", True, machine=MACHINE) is None
    other_gpu = {**MACHINE, "gpu": "Another GPU"}
    assert vc.lookup("org/m-1B", "lora", True, machine=other_gpu) is None
    new_torch = {**MACHINE, "torch": "2.11"}
    assert vc.lookup("org/m-1B", "lora", True, machine=new_torch) is None


def test_lookup_survives_a_missing_corrupt_or_old_store(store):
    assert vc.lookup("org/m-1B", machine=MACHINE) is None
    store.write_text("{not json", encoding="utf-8")
    assert vc.lookup("org/m-1B", machine=MACHINE) is None
    vc.save(_cal(version=0))
    assert vc.lookup("org/m-1B", machine=MACHINE) is None  # older schema: ignored


def test_no_gpu_means_no_calibration(store, monkeypatch):
    vc.save(_cal())
    monkeypatch.setattr(vc, "machine_fingerprint", lambda: None)
    assert vc.lookup("org/m-1B") is None


# ---- estimate_vram uses a measurement -------------------------------------------------

_SHAPE = {"param_count_billions": 1.0, "hidden_dim": 2048, "num_layers": 16,
          "num_heads": 32, "vocab_size": 128256}
_QV = {"lora_r": 16, "target_modules": "q_proj,v_proj"}


def _use(monkeypatch, cal):
    monkeypatch.setattr(vc, "lookup", lambda model, mode="lora", base_4bit=True, machine=None: cal)


def test_estimate_prefers_the_measurement(monkeypatch):
    cal = _cal(probe_trainable_params=0)
    _use(monkeypatch, cal)
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=4, overhead_fraction=0.0)
    assert est.source == "measured"
    trainable = 16 * (3 * 2048 + 2048 * 8 // 32) * 16  # q + v at kv_heads = heads / 4
    expected = 1.0 + trainable * (4 + 6.3) / GIB + cal.rows_gib(4, 2048)
    assert est.total_gb == pytest.approx(expected, rel=1e-6)
    assert any("measured on this GPU" in n for n in est.notes)


def test_floor_applies_to_small_batches(monkeypatch):
    _use(monkeypatch, _cal(floor_gib=3.0, probe_trainable_params=0))
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=1, max_seq_length=256,
                        overhead_fraction=0.0)
    assert est.activations_gb == pytest.approx(3.0)


def test_unmeasured_rows_fall_back_to_the_formula_rows(monkeypatch):
    plain = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=4, overhead_fraction=0.0,
                          use_calibration=False)
    _use(monkeypatch, _cal(quad_bytes=None, lin_bytes=None, floor_gib=0.1))
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=4, overhead_fraction=0.0)
    assert est.source == "measured"
    assert est.activations_gb == pytest.approx(plain.activations_gb)
    assert est.model_weights_gb == 1.0
    assert any("per-row cost from the formula" in n for n in est.notes)


@pytest.mark.parametrize(
    "kwargs",
    [{"use_calibration": False}, {"gradient_checkpointing": False}, {"varlen_attention": False}],
)
def test_cases_that_keep_the_formula(monkeypatch, kwargs):
    _use(monkeypatch, _cal())
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=2, **kwargs)
    assert est.source == "estimate"


def test_formula_has_the_embedding_floor():
    est = estimate_vram("org/x", **_SHAPE, **_QV, batch_size=1, max_seq_length=256,
                        varlen_attention=False, overhead_fraction=0.0)
    assert est.activations_gb == pytest.approx(4 * 128256 * 2048 / GIB)  # 0.98 GiB measured
    assert est.source == "estimate"


# ---- the parent: reading the child ---------------------------------------------------


class _FakeProc:
    def __init__(self, lines, rc=0):
        self.stdout = io.StringIO("\n".join(lines) + "\n")
        self._rc = rc

    def wait(self):
        return self._rc

    def kill(self):
        pass


def _run_calibrate(monkeypatch, store, rows, rc=0):
    lines = ["noise from the trainer", *("CALIBRATION " + json.dumps(r) for r in rows)]
    monkeypatch.setattr(vc, "machine_fingerprint", lambda: MACHINE)
    monkeypatch.setattr(vc.subprocess, "Popen", lambda *a, **k: _FakeProc(lines, rc))
    events = []
    cal = vc.calibrate("org/m-1B", on_event=events.append)
    return cal, events


def test_calibrate_fits_stores_and_reports_progress(monkeypatch, store):
    probes = [_probe(2, 2048, 512.0, 70_000.0), _probe(8, 1024, 512.0, 70_000.0)]
    cal, events = _run_calibrate(monkeypatch, store, [
        {"event": "loading", "budget_gib": 20.0},
        {"event": "loaded", "load_gib": 1.06, "floor_gib": 0.98},
        {"event": "probe", **probes[0]},
        {"event": "probe", **probes[1]},
        {"event": "result", "load_gib": 1.06, "floor_gib": 0.98, "trainable_params": 851_968,
         "formula_quad": 512.0, "formula_lin": 71_680.0, "probes": probes},
    ])
    assert [e["event"] for e in events] == ["loading", "loaded", "probe", "probe"]
    assert cal.quad_bytes == pytest.approx(512.0, rel=1e-3)
    assert cal.machine == MACHINE and cal.probe_trainable_params == 851_968
    assert vc.lookup("org/m-1B", machine=MACHINE).floor_gib == 0.98  # stored


def test_calibrate_without_informative_probes_still_stores_the_load(monkeypatch, store):
    cal, _events = _run_calibrate(monkeypatch, store, [
        {"event": "result", "load_gib": 6.7, "floor_gib": 2.03, "trainable_params": 1,
         "formula_quad": 448.0, "formula_lin": 125_440.0, "probes": []},
    ])
    assert cal.rows_measured is False and cal.load_gib == 6.7
    assert vc.lookup("org/m-1B", machine=MACHINE) is not None


def test_calibrate_raises_on_a_child_error_or_no_result(monkeypatch, store):
    with pytest.raises(vc.CalibrationError, match="Nothing was run"):
        _run_calibrate(monkeypatch, store, [{"event": "result", "error": "Nothing was run."}])
    with pytest.raises(vc.CalibrationError, match="without a result"):
        _run_calibrate(monkeypatch, store, [{"event": "loading"}], rc=1)
    assert not store.exists()


def test_calibrate_stops_a_silent_child(monkeypatch, store):
    """A child that hangs without printing anything is killed by the timer."""
    import threading

    class _Hung:
        def __init__(self):
            self._killed = threading.Event()
            self.stdout = self

        def __iter__(self):
            self._killed.wait(10)  # no output until killed
            return iter(())

        def wait(self):
            return -9

        def kill(self):
            self._killed.set()

    hung = _Hung()
    monkeypatch.setattr(vc, "machine_fingerprint", lambda: MACHINE)
    monkeypatch.setattr(vc.subprocess, "Popen", lambda *a, **k: hung)
    with pytest.raises(vc.CalibrationError, match="did not finish"):
        vc.calibrate("org/m-1B", timeout_s=0.2)
    assert hung._killed.is_set() and not store.exists()


def test_calibrate_needs_a_gpu(monkeypatch, store):
    monkeypatch.setattr(vc, "machine_fingerprint", lambda: None)
    with pytest.raises(vc.CalibrationError, match="No CUDA GPU"):
        vc.calibrate("org/m-1B")


# ---- the CLI --------------------------------------------------------------------------


def _json_out(capsys):
    out = capsys.readouterr().out
    decoder, i = json.JSONDecoder(), 0
    while (i := out.find("{", i)) != -1:
        try:
            obj, _end = decoder.raw_decode(out, i)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and ("calibration" in obj or "per_config_estimate" in obj):
            return obj
        i += 1
    raise AssertionError(out)


def test_cli_calibrate_prints_and_saves(monkeypatch, capsys):
    from backpropagate.cli import main

    seen = {}

    def fake(model, *, mode, base_4bit, on_event):
        seen.update(model=model, mode=mode, base_4bit=base_4bit)
        on_event({"event": "loaded", "load_gib": 1.06, "floor_gib": 0.98})
        on_event({"event": "probe", "batch": 2, "seq": 2048, "peak_gib": 5.3})
        on_event({"event": "skipped", "batch": 4, "seq": 2048, "predicted_gib": 30.0})
        return _cal()

    monkeypatch.setattr(vc, "calibrate", fake)
    assert main(["estimate-vram", "org/m-1B", "--calibrate", "--no-4bit"]) == 0
    out = capsys.readouterr().out
    assert seen == {"model": "org/m-1B", "mode": "lora", "base_4bit": False}
    assert "Measured on Test GPU" in out and "skipped" in out
    assert main(["estimate-vram", "org/m-1B", "--calibrate", "--json"]) == 0
    assert _json_out(capsys)["calibration"]["load_gib"] == 1.0


def test_cli_calibrate_failure_is_a_coded_runtime_error(monkeypatch, capsys):
    from backpropagate.cli import main

    def boom(*a, **k):
        raise vc.CalibrationError("No CUDA GPU detected: there is nothing to measure on.")

    monkeypatch.setattr(vc, "calibrate", boom)
    assert main(["estimate-vram", "org/m-1B", "--calibrate"]) == 2
    captured = capsys.readouterr()
    assert "RUNTIME_VRAM_CALIBRATION_FAILED" in captured.out + captured.err


def test_cli_estimate_reports_the_source_and_can_ignore_a_measurement(monkeypatch, capsys):
    from backpropagate.cli import main

    _use(monkeypatch, _cal())
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    args = ["estimate-vram", "org/m-1B", "--lora-r", "16", "--batch-size", "2", "--json"]
    # --vram-gb simulates another card: a measurement made here does not apply.
    assert main([*args, "--vram-gb", "24"]) == 0
    assert _json_out(capsys)["per_config_estimate"]["source"] == "estimate"


# ---- the UI job ------------------------------------------------------------------------


def test_ui_calibrate_job_argv_and_validation(tmp_path):
    from backpropagate import ui_jobs
    from backpropagate.ui_jobs import JobSpec, JobValidationError

    spec = JobSpec(kind="calibrate", model="org/m-1B", mode="lora", base_4bit=False)
    ui_jobs._validate_spec(spec)  # no dataset needed
    argv = ui_jobs._build_argv(spec, tmp_path / "run")
    assert argv[3:5] == ["estimate-vram", "--calibrate"]
    assert argv[-2:] == ["--", "org/m-1B"]  # the positional goes last, after "--"
    assert "--no-4bit" in argv and argv[argv.index("--mode") + 1] == "lora"
    assert argv[argv.index("--ui-run-dir") + 1] == str(tmp_path / "run")
    with pytest.raises(JobValidationError):
        ui_jobs._validate_spec(JobSpec(kind="calibrate", model="not a model id"))


def test_ui_verdict_carries_the_source(monkeypatch):
    from backpropagate.ui_jobs import vram_verdict

    assert vram_verdict("org/m-1B", lora_r=16, batch="2", card_gb=24.0)["source"] == "estimate"
    _use(monkeypatch, _cal())
    assert vram_verdict("org/m-1B", lora_r=16, batch="2", card_gb=24.0)["source"] == "measured"


def test_ui_state_calibrate_labels():
    pytest.importorskip("reflex")
    from backpropagate import ui_state as us

    s = us.TrainState()
    s.job_kind = "calibrate"
    assert s.job_chip_label == "MEASURING"
    assert s.job_has_steps is False and s.job_is_training is False
    s.vram_est_source = "measured"
    assert s.vram_est_heading == "VRAM · measured on this GPU" and s.vram_est_measured is True
    assert s.vram_est_detail.startswith("Measured on this GPU for")
