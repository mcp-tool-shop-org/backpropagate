# UI job runner hardening (external review, Lane A, 2026-10-02).
"""One test per finding of the job-runner review.

The threat is an authenticated remote client (``--share`` / ``--auth``) or a
local process that can write inside the user's profile. Each test names the
finding it closes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backpropagate import ui_jobs
from backpropagate.ui_jobs import JobManager, JobRefusedError, JobSpec, JobValidationError


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A sandbox at tmp/ui-outputs with a dataset in it; ``outside`` is not in it."""
    import backpropagate.ui_security as sec

    sandbox = tmp_path / "ui-outputs"
    sandbox.mkdir()
    monkeypatch.setattr(sec, "get_ui_output_dir", lambda: sandbox)
    monkeypatch.setattr(ui_jobs, "_hf_cache_root", lambda: None)
    data = sandbox / "d.jsonl"
    data.write_text('{"text": "hi"}\n', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    return sandbox, data, outside


def _spec(data, **over):
    base = {"kind": "sft", "model": "org/tiny", "dataset_path": str(data)}
    base.update(over)
    return JobSpec(**base)


class _Proc:
    pid = 4321

    def __init__(self, alive=True):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        return 0


def _manager(tmp_path, proc=None, **kw):
    captured: dict = {}

    def spawn(argv, **popen):
        captured["argv"] = argv
        captured["env"] = popen.get("env")
        return proc or _Proc()

    return JobManager(spawn=spawn, **kw), captured


# ---- model confinement ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sft", "multi_run", "calibrate"])
def test_a_local_model_outside_the_sandbox_is_refused(box, kind):
    _sandbox, data, outside = box
    with pytest.raises(JobValidationError, match="inside the UI output folder"):
        ui_jobs._validate_spec(_spec(data, kind=kind, model=str(outside)))


def test_a_local_model_inside_the_sandbox_is_resolved_into_the_spec(box):
    sandbox, data, _outside = box
    model_dir = sandbox / "models" / "mine"
    model_dir.mkdir(parents=True)
    spec = _spec(data, model=str(sandbox / "models" / ".." / "models" / "mine"))
    ui_jobs._validate_spec(spec)
    assert spec.model == str(model_dir.resolve())


def test_a_local_model_in_the_hugging_face_cache_is_allowed(box, monkeypatch, tmp_path):
    _sandbox, data, _outside = box
    cache = tmp_path / "hf-cache"
    snapshot = cache / "models--org--m" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(ui_jobs, "_hf_cache_root", lambda: cache.resolve())
    spec = _spec(data, model=str(snapshot))
    ui_jobs._validate_spec(spec)
    assert spec.model == str(snapshot.resolve())


@pytest.mark.parametrize("model", ["../secrets", "-org/name", ".hidden/name", "org/-name", "a/b/c"])
def test_model_ids_that_are_not_org_slash_name_are_refused(box, model):
    _sandbox, data, _outside = box
    with pytest.raises(JobValidationError):
        ui_jobs._validate_spec(_spec(data, model=model))


# ---- argv: nothing a client sends can become a flag -----------------------------------


@pytest.mark.parametrize(
    "field",
    ["model", "dataset_path", "target_modules", "run_name", "mode", "merge", "method", "batch"],
)
def test_text_fields_cannot_start_with_a_dash(box, field):
    _sandbox, data, _outside = box
    with pytest.raises(JobValidationError, match="cannot start with '-'"):
        ui_jobs._validate_spec(_spec(data, **{field: "--x"}))


def test_export_and_calibrate_positionals_follow_a_double_dash(box, tmp_path):
    sandbox, _data, _outside = box
    adapter = sandbox / "adapter"
    adapter.mkdir()
    export = JobSpec(kind="export", source_path=str(adapter), export_format="lora")
    ui_jobs._validate_spec(export)
    argv = ui_jobs._build_argv(export, tmp_path / "run")
    assert argv[-2:] == ["--", str(adapter.resolve())]
    cal = ui_jobs._build_argv(JobSpec(kind="calibrate", model="org/m"), tmp_path / "run")
    assert cal[-2:] == ["--", "org/m"]


def test_the_real_parser_reads_text_after_the_double_dash_as_the_positional():
    from backpropagate.cli import create_parser

    parser = create_parser()
    export = parser.parse_args(["export", "--format", "lora", "--", "--push-to-hub=evil/repo"])
    assert getattr(export, "push_to_hub", None) in (None, False)
    cal = parser.parse_args(["estimate-vram", "--calibrate", "--", "--log-file=x"])
    assert cal.model == "--log-file=x" and getattr(cal, "log_file", None) is None


def test_validated_paths_reach_argv_resolved(box, tmp_path):
    sandbox, data, _outside = box
    spec = _spec(data, dataset_path=str(sandbox / "." / "d.jsonl"))
    ui_jobs._validate_spec(spec)
    argv = ui_jobs._build_argv(spec, tmp_path / "run")
    assert argv[argv.index("--data") + 1] == str(data.resolve())


# ---- type confusion ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("steps", True, "whole number"),
        ("steps", ["9"], "whole number"),
        ("steps", "9" * 5000, "whole number"),
        ("runs", {"a": 1}, "whole number"),
        ("lora_r", 8.5, "whole number"),
        ("samples", "40", "whole number"),
        ("lr", "0.1", "must be a number"),
        ("lr", True, "must be a number"),
        ("gpu_max_temp", [90], "must be a number"),
        ("base_4bit", "yes", "true or false"),
        ("model", ["org/tiny"], "must be text"),
        ("run_name", "x" * 500, "too long"),
        ("method_params", [1, 2], "small mapping"),
        ("method_params", {"orpo_beta": "big"}, "names to numbers"),
        ("output_dir", 7, "must be text"),
    ],
)
def test_wrong_types_are_refused_with_a_short_message(box, field, value, needle):
    _sandbox, data, _outside = box
    with pytest.raises(JobValidationError, match=needle) as err:
        ui_jobs._validate_spec(_spec(data, **{field: value}))
    assert len(str(err.value)) < 300  # the value is never echoed at length


def test_trust_remote_code_is_refused_for_every_kind(box):
    sandbox, data, _outside = box
    adapter = sandbox / "adapter"
    adapter.mkdir()
    specs = [
        _spec(data, trust_remote_code=True),
        _spec(data, kind="multi_run", trust_remote_code=True),
        JobSpec(kind="export", source_path=str(adapter), trust_remote_code=True),
        JobSpec(kind="calibrate", model="org/m", trust_remote_code=True),
    ]
    for spec in specs:
        with pytest.raises(JobValidationError, match="trust_remote_code"):
            ui_jobs._validate_spec(spec)


# ---- the jobs folder: a link cannot redirect it --------------------------------------------


def _link_dir(link: Path, target: Path) -> bool:
    """A directory symlink (POSIX) or junction (Windows). False if unavailable."""
    try:
        if os.name == "nt":
            done = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True, check=False,
            )
            return done.returncode == 0
        link.symlink_to(target, target_is_directory=True)
        return True
    except OSError:
        return False


def test_a_link_at_the_jobs_folder_cannot_redirect_job_files(box, tmp_path):
    sandbox, data, outside = box
    if not _link_dir(sandbox / "jobs", outside):
        pytest.skip("cannot create a directory link here")
    manager, captured = _manager(tmp_path)  # default root: <sandbox>/jobs
    with pytest.raises(JobValidationError, match="only writes inside"):
        manager.start(_spec(data))
    assert list(outside.iterdir()) == []  # nothing was created at the target
    assert "argv" not in captured  # and nothing was spawned


def test_the_default_jobs_folder_is_used_resolved(box, tmp_path):
    sandbox, data, _outside = box
    manager, captured = _manager(tmp_path)
    job = manager.start(_spec(data))
    assert job.run_dir.parent == (sandbox / "jobs").resolve()
    argv = captured["argv"]
    assert argv[argv.index("--ui-run-dir") + 1] == str(job.run_dir)


# ---- the child's environment -------------------------------------------------------------------


def test_the_child_does_not_inherit_the_ui_auth_secrets(box, tmp_path, monkeypatch):
    _sandbox, data, _outside = box
    monkeypatch.setenv("BACKPROPAGATE_UI_LAUNCH_TOKEN", "tok")
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH_VERIFIER", "scrypt$...")
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH_USER", "admin")
    monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "admin:pw")
    monkeypatch.setenv("HF_TOKEN", "hf_keep")
    manager, captured = _manager(tmp_path)
    manager.start(_spec(data))
    env = captured["env"]
    assert not [k for k in env if k.startswith("BACKPROPAGATE_UI_AUTH")]
    assert "BACKPROPAGATE_UI_LAUNCH_TOKEN" not in env
    assert env["HF_TOKEN"] == "hf_keep"  # the run still needs this one
    assert env["PYTHONUNBUFFERED"] == "1"


# ---- one job at a time, until the process is gone ------------------------------------------------


def test_the_slot_stays_busy_until_the_process_exits(box, tmp_path):
    _sandbox, data, _outside = box
    proc = _Proc(alive=True)
    manager, _captured = _manager(tmp_path, proc=proc)
    job = manager.start(_spec(data))
    # The job wrote its terminal event but is still tearing down (GPU held).
    with open(job.events_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "done", "status": "done"}) + "\n")
    with pytest.raises(JobRefusedError, match="already running"):
        manager.start(_spec(data))
    proc._alive = False
    manager.start(_spec(data))  # the process is gone: the slot is free


def test_a_slow_preflight_does_not_hold_the_lock(box, tmp_path, monkeypatch):
    _sandbox, data, _outside = box
    manager, _captured = _manager(tmp_path)
    seen = {}

    def preflight(spec):
        seen["locked"] = manager._lock.locked()
        return True, "fits"

    monkeypatch.setattr(ui_jobs, "_vram_preflight", preflight)
    manager.start(_spec(data))
    assert seen["locked"] is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX: the Job Object covers this on Windows")
def test_a_job_from_an_earlier_ui_session_blocks_a_new_start(box, tmp_path):
    sandbox, data, _outside = box
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "backpropagate-job"],
        start_new_session=True,
    )
    try:
        old = sandbox / "jobs" / "run_old"
        old.mkdir(parents=True)
        (old / ui_jobs.JOB_FILENAME).write_text(
            json.dumps({"job_id": "run_old", "pid": sleeper.pid, "status": "running"}),
            encoding="utf-8",
        )
        manager, _captured = _manager(tmp_path)  # a fresh manager: no memory of it
        with pytest.raises(JobRefusedError, match="earlier UI session"):
            manager.start(_spec(data))
    finally:
        sleeper.kill()
        sleeper.wait(timeout=10)
    time.sleep(0.2)
    manager.start(_spec(data))  # the old process is gone: a new job may start


def test_a_stale_running_record_with_a_dead_process_does_not_block(box, tmp_path):
    sandbox, data, _outside = box
    old = sandbox / "jobs" / "run_dead"
    old.mkdir(parents=True)
    (old / ui_jobs.JOB_FILENAME).write_text(
        json.dumps({"job_id": "run_dead", "pid": 2**22 + 12345, "status": "running"}),
        encoding="utf-8",
    )
    manager, _captured = _manager(tmp_path)
    manager.start(_spec(data))


# ---- event files: bounded reads ---------------------------------------------------------------------


def test_event_reads_are_bounded_and_resume_from_the_offset(box, tmp_path, monkeypatch):
    _sandbox, data, _outside = box
    manager, _captured = _manager(tmp_path)
    job = manager.start(_spec(data))
    rows = [{"kind": "step", "step": i, "total_steps": 500} for i in range(1, 501)]
    with open(job.events_path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps({"kind": "done", "status": "done"}) + "\n")
    monkeypatch.setattr(ui_jobs, "MAX_EVENT_READ_BYTES", 2048)
    first, offset = manager.tail_events(job, 0)
    assert 0 < len(first) < 100 and offset <= 2048  # one slice, not the whole file
    steps = [r["step"] for r in manager._iter_events(job) if r.get("kind") == "step"]
    assert steps == list(range(1, 501))  # every row, a slice at a time
    monkeypatch.setattr(ui_jobs, "_EVENT_TAIL_BYTES", 512)
    assert manager._terminal_seen(job) is True  # found in the tail window
    assert manager._terminal_status(job) == "done"
    assert len(manager._read_events(job, limit=3)) == 3
    assert manager.status()["status"] == "active"  # terminal event, process still exiting
    manager._jobs[job.job_id]._alive = False
    assert manager.status()["status"] == "done"


def test_a_hostile_event_file_is_tolerated(box, tmp_path):
    _sandbox, data, _outside = box
    proc = _Proc(alive=True)
    manager, _captured = _manager(tmp_path, proc=proc)
    job = manager.start(_spec(data))
    with open(job.events_path, "ab") as fh:
        fh.write(b"\xff\xfe not json at all\n")
        fh.write(b'["a", "list", "not", "an", "object"]\n')
        fh.write(b'{"kind": "step", "step": 3}\n')
    rows, _offset = manager.tail_events(job, 0)
    assert any(isinstance(r, dict) and r.get("step") == 3 for r in rows)
    assert manager.status()["status"] == "active"


# ---- a job is active until its process has exited -------------------------------------


def _finish(job, **row):
    with open(job.events_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    ("row", "final"),
    [
        ({"kind": "done", "status": "done"}, "done"),
        ({"kind": "done", "status": "stopped"}, "stopped"),
        ({"kind": "error", "code": "RUNTIME_X", "message": "boom"}, "failed"),
    ],
)
def test_a_finished_job_reads_active_until_its_process_exits(box, tmp_path, row, final):
    """start() refuses a new job until the old process is gone. Reporting the
    job finished at its terminal event let the page offer Start for a moment
    and then refuse the click."""
    _sandbox, data, _outside = box
    proc = _Proc(alive=True)
    manager, _captured = _manager(tmp_path, proc=proc)
    job = manager.start(_spec(data))
    _finish(job, **row)
    status = manager.status()
    assert (status["status"], status["finishing"]) == ("active", True)
    with pytest.raises(JobRefusedError):
        manager.start(_spec(data))
    proc._alive = False
    status = manager.status()
    assert (status["status"], status["finishing"]) == (final, False)


def test_a_process_that_never_exits_does_not_hold_the_page_forever(box, tmp_path, monkeypatch):
    _sandbox, data, _outside = box
    manager, _captured = _manager(tmp_path, proc=_Proc(alive=True))
    job = manager.start(_spec(data))
    _finish(job, kind="done", status="done")
    assert manager.status()["status"] == "active"
    clock = [time.monotonic()]
    monkeypatch.setattr(ui_jobs.time, "monotonic", lambda: clock[0])
    clock[0] += ui_jobs.EXIT_GRACE_S - 1
    assert manager.status()["status"] == "active"
    clock[0] += 2
    status = manager.status()
    assert (status["status"], status["finishing"]) == ("done", False)


def test_a_running_job_is_not_finishing(box, tmp_path):
    _sandbox, data, _outside = box
    manager, _captured = _manager(tmp_path, proc=_Proc(alive=True))
    job = manager.start(_spec(data))
    _finish(job, kind="step", step=3, total_steps=10, loss=1.0)
    status = manager.status()
    assert (status["status"], status["finishing"]) == ("active", False)
