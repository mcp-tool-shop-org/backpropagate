# VRAM calibration: the branches the main test file does not reach.
"""Edge and wiring coverage for ``backpropagate.vram_calibration`` and the
CLI / UI code around it: the machine fingerprint, store corner cases, the
helpers the GPU child uses (run here without a GPU), every progress event in
the CLI body, the UI job handlers, and the estimate's fallbacks.
"""

from __future__ import annotations

import argparse
import io
import json
from types import SimpleNamespace

import pytest

from backpropagate import vram_calibration as vc
from backpropagate.trainer import estimate_vram

MACHINE = {"gpu": "Test GPU", "vram_gib": 24.0, "torch": "2.10", "transformers": "5.5",
           "unsloth": "2026.5", "attention": "sdpa-efficient"}


def _cal(**kw) -> vc.Calibration:
    base = {
        "model": "org/m-1B", "mode": "lora", "base_4bit": True, "machine": MACHINE,
        "load_gib": 1.0, "floor_gib": 1.0, "quad_bytes": 500.0, "lin_bytes": 70_000.0,
        "probe_trainable_params": 1_000_000, "max_residual_pct": 1.0,
        "measured_at": "2026-10-02T10:00:00",
    }
    base.update(kw)
    return vc.Calibration(**base)


# ---- module helpers -----------------------------------------------------------------


def test_default_store_path_is_in_the_home_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("BACKPROPAGATE_VRAM_CALIBRATION", raising=False)
    monkeypatch.setattr(vc.Path, "home", classmethod(lambda cls: tmp_path))
    assert vc.calibration_path() == tmp_path / ".backpropagate" / "vram-calibration.json"


def test_lib_version_reads_installed_and_missing_packages():
    assert vc._lib_version("pytest") != ""
    assert vc._lib_version("definitely-not-a-real-package-xyz") == ""


def test_machine_fingerprint_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert vc.machine_fingerprint() is None


def test_machine_fingerprint_with_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda index: SimpleNamespace(total_memory=24 * vc.GIB),
    )
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    fp = vc.machine_fingerprint()
    assert fp is not None
    assert fp["gpu"] == "Test GPU" and fp["vram_gib"] == 24.0
    assert set(fp) == {"gpu", "vram_gib", "torch", "transformers", "unsloth", "attention"}
    assert fp["attention"] in ("varlen", "sdpa-efficient")


def test_machine_fingerprint_survives_a_broken_cuda(monkeypatch):
    def boom():
        raise RuntimeError("driver gone")

    monkeypatch.setattr("torch.cuda.is_available", boom)
    assert vc.machine_fingerprint() is None


def test_lookup_ignores_an_entry_with_unknown_fields(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    monkeypatch.setenv("BACKPROPAGATE_VRAM_CALIBRATION", str(path))
    vc.save(_cal())
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.values():
        entry["a_field_from_the_future"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    assert vc.lookup("org/m-1B", machine=MACHINE) is None


def test_fit_returns_none_when_no_probe_is_usable():
    assert vc.fit_rows([{"batch": 2, "seq": 2048, "overhead_gib": 0.0}]) is None
    assert vc.fit_rows([{"batch": 0, "seq": 2048, "overhead_gib": 1.0}]) is None
    assert vc.fit_rows([{"batch": 2, "seq": 2048, "overhead_gib": 1.0, "oom": True}]) is None


def test_a_calibration_stored_by_the_older_fit_is_ignored(monkeypatch, tmp_path):
    path = tmp_path / "cal.json"
    monkeypatch.setenv("BACKPROPAGATE_VRAM_CALIBRATION", str(path))
    vc.save(_cal())
    assert vc.lookup("org/m-1B", machine=MACHINE) is not None
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.values():
        entry["version"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    assert vc.lookup("org/m-1B", machine=MACHINE) is None


def test_probe_dataset_rows_are_longer_than_the_row_length(tmp_path):
    path = vc._probe_dataset(tmp_path, 256, 3)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert all(len(r["messages"][1]["content"].split()) > 256 for r in rows)


def test_emit_and_oom_detection(capsys):
    vc._emit({"event": "loaded", "load_gib": 1.0})
    line = capsys.readouterr().out.strip()
    assert line.startswith("CALIBRATION ") and json.loads(line[12:])["event"] == "loaded"
    assert vc._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 8.00 GiB"))
    assert vc._is_oom(MemoryError("RUNTIME_GPU_OOM"))
    assert not vc._is_oom(ValueError("bad dataset"))


class _FakeProc:
    def __init__(self, lines):
        self.stdout = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))

    def wait(self):
        return 0

    def poll(self):
        return 0

    def kill(self):
        pass


def test_calibrate_reads_bytes_and_skips_a_malformed_line(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKPROPAGATE_VRAM_CALIBRATION", str(tmp_path / "cal.json"))
    monkeypatch.setattr(vc, "machine_fingerprint", lambda: MACHINE)
    lines = [
        "CALIBRATION {not json",
        "CALIBRATION " + json.dumps({"event": "result", "load_gib": 2.0, "floor_gib": 0.5,
                                     "trainable_params": 10, "formula_quad": 400.0,
                                     "formula_lin": 50_000.0, "probes": []}),
    ]
    monkeypatch.setattr(vc.subprocess, "Popen", lambda *a, **k: _FakeProc(lines))
    cal = vc.calibrate("org/m-1B")  # no on_event: progress rows are dropped
    assert cal.load_gib == 2.0 and cal.rows_measured is False


# ---- the CLI body: every progress event, with and without a UI job ---------------------


class _Writer:
    def __init__(self):
        self.phases = []

    def phase(self, name, **extra):
        self.phases.append(name)


def _args(**kw):
    base = {"model": "org/m-1B", "mode": "lora", "no_4bit": False, "json": False}
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_calibrate(result):
    def run(model, *, mode, base_4bit, on_event):
        on_event({"event": "loading", "budget_gib": 20.0})
        on_event({"event": "loaded", "load_gib": 1.06, "floor_gib": 0.98})
        on_event({"event": "probe", "batch": 2, "seq": 2048, "peak_gib": 5.3})
        on_event({"event": "skipped", "batch": 8, "seq": 2048, "predicted_gib": 30.0})
        on_event({"event": "probe", "batch": 4, "seq": 2048, "oom": True})
        return result

    return run


def test_cli_body_reports_every_event_and_feeds_the_ui_job(monkeypatch, capsys):
    from backpropagate import cli

    monkeypatch.setattr(vc, "calibrate", _fake_calibrate(_cal()))
    job = SimpleNamespace(writer=_Writer(), output_path=None)
    assert cli._cmd_calibrate_body(_args(_ui_job=job, mode="full")) == 0
    out = capsys.readouterr().out
    assert "Loading org/m-1B" in out and "ran out of memory" in out and "skipped" in out
    assert "Mode: full" in out.replace("  ", " ")
    assert job.writer.phases == [
        "measuring", "measured batch 2 x 2048 tokens", "measured batch 4 x 2048 tokens",
    ]
    assert job.output_path is None  # the store is outside the sandbox: never the job output


def test_cli_body_says_when_only_the_load_size_was_measured(monkeypatch, capsys):
    from backpropagate import cli

    monkeypatch.setattr(vc, "calibrate", _fake_calibrate(_cal(quad_bytes=None, lin_bytes=None)))
    assert cli._cmd_calibrate_body(_args(no_4bit=True)) == 0
    out = capsys.readouterr().out
    assert "LoRA (16-bit)" in out
    assert "per-row cost stays the" in out


def test_cli_estimate_shows_a_measured_source(monkeypatch, capsys):
    from backpropagate.cli import main

    monkeypatch.setattr(vc, "lookup", lambda model, mode="lora", base_4bit=True, machine=None: _cal())
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "torch.cuda.get_device_properties",
        lambda index: SimpleNamespace(total_memory=24 * vc.GIB),
    )
    assert main(["estimate-vram", "org/m-1B", "--lora-r", "16", "--batch-size", "2"]) == 0
    assert "measured on this GPU" in capsys.readouterr().out
    assert main(["estimate-vram", "org/m-1B", "--lora-r", "16", "--batch-size", "2",
                 "--no-calibration"]) == 0
    assert "built-in formula" in capsys.readouterr().out


# ---- estimate_vram: a broken store never breaks the estimate ---------------------------


def test_estimate_survives_a_failing_lookup(monkeypatch):
    def boom(*a, **k):
        raise OSError("store unreadable")

    monkeypatch.setattr(vc, "lookup", boom)
    est = estimate_vram("org/m-1B", lora_r=16, batch_size=1)
    assert est.source == "estimate" and est.total_gb > 0


def test_measured_full_fine_tune_uses_the_measured_training_cost(monkeypatch):
    """The probes' overhead already holds gradients and optimizer state.
    Adding the formula's term as well counted them twice (about 8.6 GiB too
    much for a 3B model)."""
    shape = {"param_count_billions": 1.0, "hidden_dim": 2048, "num_layers": 16,
             "num_heads": 32, "vocab_size": 128256}
    plain = estimate_vram("org/m-1B", mode="full", batch_size=1, use_calibration=False, **shape)
    cal = _cal(mode="full", base_4bit=False, load_gib=2.5, fixed_gib=5.1, quad_bytes=0.0)
    monkeypatch.setattr(vc, "lookup", lambda model, mode="lora", base_4bit=True, machine=None: cal)
    est = estimate_vram("org/m-1B", mode="full", batch_size=1, **shape)
    assert est.source == "measured" and est.model_weights_gb == 2.5
    assert est.optimizer_state_gb == 5.1
    assert plain.optimizer_state_gb != pytest.approx(5.1)
    assert est.lora_adapter_gb == 0.0


# ---- the UI: job spec, handlers and labels ----------------------------------------------


def test_calibrate_spec_validation_and_default_argv(tmp_path):
    from backpropagate import ui_jobs
    from backpropagate.ui_jobs import JobSpec, JobValidationError

    argv = ui_jobs._build_argv(JobSpec(kind="calibrate", model="org/m-1B"), tmp_path / "run")
    assert "--no-4bit" not in argv
    full = ui_jobs._build_argv(
        JobSpec(kind="calibrate", model="org/m-1B", mode="full", base_4bit=False), tmp_path / "run"
    )
    assert "--no-4bit" not in full and full[full.index("--mode") + 1] == "full"
    with pytest.raises(JobValidationError, match="empty"):
        ui_jobs._validate_spec(JobSpec(kind="calibrate", model=""))
    with pytest.raises(JobValidationError, match="Unknown mode"):
        ui_jobs._validate_spec(JobSpec(kind="calibrate", model="org/m-1B", mode="bogus"))


@pytest.fixture
def train_state():
    pytest.importorskip("reflex")
    from backpropagate import ui_state as us

    return us, us.TrainState()


def test_start_calibration_builds_the_spec_from_the_form(train_state, monkeypatch):
    us, s = train_state
    seen = {}
    monkeypatch.setattr(us.TrainState, "_begin_job", lambda self, spec: seen.setdefault("spec", spec))
    s.model = "org/m-1B"
    s.train_mode = "lora"
    s.start_calibration()
    spec = seen["spec"]
    assert (spec.kind, spec.model, spec.mode, spec.base_4bit) == ("calibrate", "org/m-1B", "lora", False)
    s.train_mode = "full"
    seen.clear()
    s.start_calibration()
    assert seen["spec"].mode == "full"


def test_start_calibration_refuses_a_bad_model_field(train_state, monkeypatch):
    us, s = train_state
    monkeypatch.setattr(us.TrainState, "_begin_job", lambda self, spec: pytest.fail("started"))
    s.model_error = "Model path must be inside the UI output folder"
    s.start_calibration()
    assert "Fix the model field first" in s.job_refusal


def test_finished_calibration_refreshes_the_estimate_and_labels(train_state, monkeypatch):
    us, s = train_state
    from backpropagate import ui_jobs

    monkeypatch.setattr(
        ui_jobs, "vram_verdict",
        lambda model, **kw: {"verdict": "fits", "total_gb": 4.8, "batch": 6, "note": "",
                             "source": "measured", "seen": kw},
    )
    s.job_kind = "calibrate"
    s.vram_total_gb = 31.8
    s._finalize_job({"status": "done"})
    assert s.run_state == "done" and s.done_title == "Measured on this GPU"
    assert s.vram_est_source == "measured" and s.vram_est_total == 4.8 and s.vram_est_batch == 6
    assert "this GPU's own numbers" in s.events[-1]["msg"]
    assert s.job_is_training is False
    s._finalize_job({"status": "stopped"})
    assert s.done_title == "Measurement stopped" and s.events[-1]["msg"] == "Measurement cancelled."
    s._finalize_job({"status": "failed"})
    assert "could not be completed" in s.events[-1]["msg"]


def test_estimate_args_follow_the_form(train_state):
    _us, s = train_state
    s.train_mode = "lora"
    s.lora_r = 16
    s.batch_size = "2"
    s.target_modules = "q_proj, v_proj"
    s.gradient_checkpointing = False
    s.vram_total_gb = 24.0
    assert s._estimate_args() == {
        "mode": "lora", "lora_r": 16, "batch": "2", "base_4bit": False,
        "gradient_checkpointing": False, "card_gb": 24.0, "target_modules": "q_proj,v_proj",
        "method": "sft",
    }


def test_stop_cancels_a_running_calibration(train_state, monkeypatch):
    us, s = train_state
    from backpropagate import ui_jobs

    calls = []
    manager = SimpleNamespace(is_alive=lambda job_id: True, cancel=calls.append)
    monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: manager)
    s.job_kind = "calibrate"
    s.job_id = "run_x"
    s.stop_training()
    assert calls == ["run_x"] and s.stop_requested is True
    assert s.events[-1]["msg"] == "Cancelling the measurement."
