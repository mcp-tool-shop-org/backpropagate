# ui-v2 P1: parent-side JobManager contract (backpropagate/ui_jobs.py).
"""JobManager lifecycle tests — fake processes, no GPU required."""

from __future__ import annotations

import json
import sys

import pytest

from backpropagate.job_events import EVENTS_FILENAME
from backpropagate.ui_jobs import (
    JOB_FILENAME,
    JobManager,
    JobRefusedError,
    JobSpec,
    JobValidationError,
    _vram_preflight,
)


class _FakeProc:
    """Process double: controllable liveness, waits instantly.

    ``dies_on``: optional Path — when that file appears, the next ``poll()``
    reports exit (models a child that honors control.json). ``on_exit``
    runs once at that transition (models the child writing its done event).
    """

    def __init__(self, alive=True, dies_on=None, on_exit=None):
        self.pid = 98765
        self._alive = alive
        self.returncode = None
        self.killed = False
        self._handle = 0
        self._dies_on = dies_on
        self._on_exit = on_exit

    def _maybe_die(self):
        if self._alive and self._dies_on is not None and self._dies_on.exists():
            self._alive = False
            if self._on_exit is not None:
                self._on_exit()

    def poll(self):
        self._maybe_die()
        if not self._alive:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode
        return None

    def wait(self, timeout=None):
        self._alive = False
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self._alive = False


class _Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        self.t += 0.001
        return self.t


def _spec(tmp_path, **over):
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "hi"}\n')
    defaults = {
        "kind": "sft",
        "model": "org/tiny-model",
        "dataset_path": str(data),
        "steps": 10,
        "lr": 1e-4,
        "lora_r": 8,
    }
    defaults.update(over)
    return JobSpec(**defaults)


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point the UI output sandbox at the test tmp dir."""
    import backpropagate.ui_security as sec

    monkeypatch.setattr(sec, "get_ui_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "backpropagate.ui_jobs.get_ui_output_dir", lambda: tmp_path, raising=False
    )
    return tmp_path


def _manager(tmp_path, proc=None, clock=None):
    fake = proc if proc is not None else _FakeProc()
    captured: dict = {}

    def spawn(argv, **kw):
        captured["argv"] = argv
        captured["env"] = kw.get("env")
        return fake

    m = JobManager(jobs_root=tmp_path / "jobs", spawn=spawn, clock=clock or _Clock())
    return m, fake, captured


def test_start_builds_run_dir_and_argv(tmp_path):
    m, _proc, captured = _manager(tmp_path)
    spec = _spec(tmp_path, scratch_root=str(tmp_path))
    job = m.start(spec)
    assert job.job_id.startswith("run_")
    assert job.run_dir.name == job.job_id
    assert job.events_path == job.run_dir / EVENTS_FILENAME
    argv = captured["argv"]
    assert argv[1:3] == ["-m", "backpropagate"]
    assert argv[3] == "train"
    assert "--ui-run-dir" in argv
    assert argv[argv.index("--ui-run-dir") + 1] == str(job.run_dir)
    assert argv[argv.index("--model") + 1] == "org/tiny-model"
    assert argv[argv.index("--steps") + 1] == "10"
    # spawn record on disk
    record = json.loads((job.run_dir / JOB_FILENAME).read_text())
    assert record["job_id"] == job.job_id
    assert record["status"] == "running"
    assert record["pid"] == job.pid


def test_refuses_second_concurrent_start(tmp_path):
    m, proc, _ = _manager(tmp_path)
    spec = _spec(tmp_path, scratch_root=str(tmp_path))
    job = m.start(spec)
    with pytest.raises(JobRefusedError):
        m.start(spec)
    # original still runs and can accumulate events
    writer_path = job.events_path
    assert not writer_path.read_text().strip().count('"kind": "done"')
    from backpropagate.job_events import JobEventWriter

    JobEventWriter(job.run_dir).phase("training")
    rows, _ = m.tail_events(job, 0)
    assert any(r.get("phase") == "training" for r in rows)
    assert m._is_alive(job.job_id)
    proc._alive = False  # cleanup state


def test_stop_on_dead_process_reports_crashed(tmp_path):
    m, proc, _ = _manager(tmp_path, proc=_FakeProc(alive=True))
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    proc._alive = False  # simulate a crash between polls
    assert m.stop_current(grace_s=0.1) == "crashed"


def test_stop_on_finished_job_reports_done(tmp_path):
    m, proc, _ = _manager(tmp_path)
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    from backpropagate.job_events import JobEventWriter

    JobEventWriter(job.run_dir).done(status="completed", steps_done=10)
    proc._alive = False
    assert m.stop_current(grace_s=0.1) in ("completed", "done")


def test_stop_writes_control_file_for_live_job(tmp_path):
    from backpropagate.job_events import JobEventWriter

    m0 = JobManager(jobs_root=tmp_path / "jobs", spawn=None, clock=_Clock())
    run_dir_holder: dict = {}

    def _child_exits():
        # the real child writes done{status:"stopped"} after the cooperative
        # stop unwinds the trainer loop
        JobEventWriter(run_dir_holder["run_dir"]).done(status="stopped", steps_done=3)

    def spawn(argv, **kw):
        run_dir = argv[argv.index("--ui-run-dir") + 1]
        run_dir_holder["run_dir"] = run_dir
        from pathlib import Path

        return _FakeProc(
            alive=True,
            dies_on=Path(run_dir) / "control.json",
            on_exit=_child_exits,
        )

    m = JobManager(jobs_root=tmp_path / "jobs", spawn=spawn, clock=_Clock())
    del m0
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    status = m.stop_current(graceful=True, grace_s=0.3)
    assert (job.run_dir / "control.json").exists()
    payload = json.loads((job.run_dir / "control.json").read_text())
    assert payload["action"] == "stop_save"
    assert status == "stopped"


def test_validation_rejects_bad_specs(tmp_path):
    m, _, _ = _manager(tmp_path)
    with pytest.raises(JobValidationError):
        m.start(_spec(tmp_path, model=""))
    with pytest.raises(JobValidationError):
        m.start(_spec(tmp_path, dataset_path=str(tmp_path / "missing.jsonl")))
    with pytest.raises(JobValidationError):
        m.start(_spec(tmp_path, steps=0))
    with pytest.raises(JobValidationError):
        m.start(_spec(tmp_path, lr=0.0))
    # ui-v2 P3: full fine-tuning is allowed from the UI, for SFT only.
    with pytest.raises(JobValidationError):
        m.start(_spec(tmp_path, mode="full", method="orpo"))
    with pytest.raises(JobValidationError):
        m.start(_spec(tmp_path, mode="bogus"))
    with pytest.raises(NotImplementedError):
        m.start(_spec(tmp_path, kind="export_gguf"))


def test_sandbox_refuses_paths_outside_ui_output(tmp_path, sandbox):
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text('{"text": "x"}\n')
    m, _, _ = _manager(tmp_path)
    with pytest.raises(JobValidationError, match="outside"):
        m.start(_spec(tmp_path, dataset_path=str(outside)))


def test_sandbox_refuses_when_unverifiable(tmp_path, monkeypatch):
    """Fix-round: the sandbox check FAILS CLOSED. A broken ui_security import
    or resolve error is a refusal, not a silent pass-through."""
    import backpropagate.ui_security as sec

    def boom():
        raise RuntimeError("ui_security broken in this harness")

    monkeypatch.setattr(sec, "get_ui_output_dir", boom)
    m, _, _ = _manager(tmp_path)
    with pytest.raises(JobValidationError, match="sandbox"):
        m.start(_spec(tmp_path))


def test_spawn_record_redacts_absolute_paths(tmp_path):
    """job.json argv_tail keeps flags but collapses absolute path values to
    <path> — no rig drive letters leak into records (fix-round #12a)."""
    m, _, _ = _manager(tmp_path)
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    record = json.loads((job.run_dir / JOB_FILENAME).read_text())
    tail = [str(a) for a in record["argv_tail"]]
    assert "--model" in tail and "org/tiny-model" in tail
    assert "<path>" in tail  # the dataset + output paths collapsed
    assert not any(a.lower().startswith(("e:", "c:", "/", "\\\\")) for a in tail)


def test_vram_preflight_quotes_gib(tmp_path, monkeypatch):
    """Preflight must use GiB (the unit the UI displays for the card), not
    decimal GB — the first cut said '32.5 GB free' on a 31.8 GiB card."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (int(30.5 * 1024**3), int(32.0 * 1024**3)),  # device-wide bytes
    )

    class _Estimate:
        total_gb = 1.0

    monkeypatch.setattr(
        "backpropagate.trainer.estimate_vram", lambda *a, **k: _Estimate()
    )
    ok, note = _vram_preflight(_spec(tmp_path))
    assert ok and note.startswith("fits:")
    assert "30.5 GB free" in note  # GiB — a /1e9 cut would have said 32.7


def test_tail_events_offsets_and_partial_lines(tmp_path):
    m, _, _ = _manager(tmp_path)
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    path = job.events_path
    path.write_text('{"kind": "phase", "phase": "loading"}\n')
    rows, offset = m.tail_events(job, 0)
    assert rows == [{"kind": "phase", "phase": "loading"}]
    assert offset > 0
    # partial trailing line held for next poll
    with open(path, "a") as fh:
        fh.write('{"kind": "step", "step": 1, "los')
    rows2, offset2 = m.tail_events(job, offset)
    assert rows2 == []
    assert offset2 == offset
    with open(path, "a") as fh:
        fh.write('s": 0.5}\n')
    rows3, offset3 = m.tail_events(job, offset2)
    assert rows3 == [{"kind": "step", "step": 1, "los": 0.5}] or rows3[0]["step"] == 1
    assert offset3 > offset2


def test_status_shape_idle_and_active(tmp_path):
    m, _, _ = _manager(tmp_path)
    assert m.status()["status"] == "idle"
    m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    st = m.status()
    assert st["status"] == "active"
    assert st["phase"] in ("queued", "training")
    for key in ("job_id", "step", "total_steps", "started_at", "log_path"):
        assert key in st


def test_grace_window_uses_last_step_time(tmp_path):
    m, _, _ = _manager(tmp_path)
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    from backpropagate.job_events import JobEventWriter

    writer = JobEventWriter(job.run_dir)
    writer.write({"kind": "step", "step": 5, "step_time_ms": 30000})
    assert m._grace_window(job, None) == max(60.0, 3 * 30 + 20)
    assert m._grace_window(job, 5.0) == 5.0


@pytest.mark.skipif(sys.platform != "win32", reason="tree kill parity check")
def test_real_child_tree_killed_after_grace_expires(tmp_path):
    """A child that ignores control.json gets tree-killed after the grace."""
    import subprocess

    proc_holder: dict = {}

    def spawn(argv, **kw):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        proc_holder["proc"] = proc
        return proc

    m = JobManager(
        jobs_root=tmp_path / "jobs",
        spawn=spawn,
        clock=_Clock(),
    )
    job = m.start(_spec(tmp_path, scratch_root=str(tmp_path)))
    assert proc_holder["proc"].poll() is None
    status = m.stop_current(grace_s=0.5)
    proc = proc_holder["proc"]
    for _ in range(40):  # taskkill runs async-ish on some rigs; poll briefly
        if proc.poll() is not None:
            break
        import time

        time.sleep(0.1)
    assert proc.poll() is not None
    assert status == "stopped"
    assert job.job_id
