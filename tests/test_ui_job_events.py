# ui-v2 P1: child-side job event contract (backpropagate/job_events.py).
"""Tests for the events.jsonl / control.json contract."""

from __future__ import annotations

import json
import time

from backpropagate.job_events import (
    CONTROL_FILENAME,
    EVENTS_FILENAME,
    JobEventWriter,
    UiFileEventCallback,
    read_stop_request,
)


def _rows(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


class _Args:
    def __init__(self, logging_steps=1, output_dir="out"):
        self.logging_steps = logging_steps
        self.output_dir = output_dir


class _State:
    def __init__(self, step=0, max_steps=10):
        self.global_step = step
        self.max_steps = max_steps
        self.epoch = 0.0


class _Control:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = False


def test_step_events_have_required_keys(tmp_path):
    cb = UiFileEventCallback(tmp_path)
    args, state, control = _Args(), _State(step=3, max_steps=20), _Control()
    cb.on_train_begin(args, state, control)
    time.sleep(0.01)  # step_time_ms depends on wall clock between logs
    cb.on_log(args, state, control, logs={"loss": 1.0, "learning_rate": 1e-4})
    cb.on_log(args, _State(step=4, max_steps=20), control, logs={"loss": 0.5, "learning_rate": 1e-4})
    steps = [r for r in _rows(tmp_path) if r.get("kind") == "step"]
    assert len(steps) == 2
    for r in steps:
        for key in (
            "ts", "kind", "step", "total_steps", "phase",
            "loss", "ema_loss", "lr", "step_time_ms",
            "vram_alloc_gib", "vram_reserved_gib", "temp_c",
        ):
            assert key in r
        assert r["phase"] == "training"
    assert steps[0]["step"] == 3
    assert steps[0]["total_steps"] == 20
    # second row carries a measured per-step time
    assert steps[1]["step_time_ms"] is not None and steps[1]["step_time_ms"] >= 0


def test_ema_is_debiased(tmp_path):
    cb = UiFileEventCallback(tmp_path)
    args, state, control = _Args(), _State(), _Control()
    cb.on_train_begin(args, state, control)
    for i, loss in enumerate([1.0, 1.0, 1.0]):
        cb.on_log(args, _State(step=i), control, logs={"loss": loss})
    steps = [r for r in _rows(tmp_path) if r.get("kind") == "step"]
    # constant series: debiased EMA converges to the constant immediately
    assert abs(steps[0]["ema_loss"] - 1.0) < 1e-9
    assert abs(steps[-1]["ema_loss"] - 1.0) < 1e-9


def test_stop_request_flips_save_and_stop(tmp_path):
    cb = UiFileEventCallback(tmp_path)
    control = _Control()
    cb.on_step_end(_Args(), _State(), control)
    assert not control.should_training_stop
    (tmp_path / CONTROL_FILENAME).write_text(json.dumps({"action": "stop_save"}))
    cb.on_step_end(_Args(), _State(), control)
    assert control.should_training_stop is True
    assert control.should_save is True


def test_malformed_control_file_is_ignored(tmp_path):
    (tmp_path / CONTROL_FILENAME).write_text("not json {")
    assert read_stop_request(tmp_path) is False
    cb = UiFileEventCallback(tmp_path)
    control = _Control()
    cb.on_step_end(_Args(), _State(), control)
    assert control.should_training_stop is False
    (tmp_path / CONTROL_FILENAME).write_text(json.dumps([1, 2, 3]))
    assert read_stop_request(tmp_path) is False


def test_done_and_checkpoint_events(tmp_path):
    writer = JobEventWriter(tmp_path)
    writer.checkpoint(tmp_path / "out" / "checkpoint-40")
    writer.done(status="completed", steps_done=40)
    rows = _rows(tmp_path)
    ckpt = next(r for r in rows if r["kind"] == "checkpoint")
    done = next(r for r in rows if r["kind"] == "done")
    for key in ("ts", "kind", "path"):
        assert key in ckpt
    for key in ("ts", "kind", "status", "steps_done"):
        assert key in done
    assert done["status"] == "completed"
    assert done["steps_done"] == 40
    assert "checkpoint-40" in ckpt["path"]


def test_error_event_carries_code_and_hint(tmp_path):
    writer = JobEventWriter(tmp_path)
    writer.error(code="DATASET_FORMAT", message="bad jsonl", hint="Use ShareGPT format")
    row = next(r for r in _rows(tmp_path) if r["kind"] == "error")
    assert row["code"] == "DATASET_FORMAT"
    assert row["hint"] == "Use ShareGPT format"
    assert row["status"] == "failed"


def test_on_log_respects_logging_steps(tmp_path):
    cb = UiFileEventCallback(tmp_path)
    args = _Args(logging_steps=10)
    control = _Control()
    cb.on_train_begin(args, _State(), control)
    cb.on_log(args, _State(step=5), control, logs={"loss": 1.0})  # off-cadence
    cb.on_log(args, _State(step=10), control, logs={"loss": 1.0})  # on-cadence
    steps = [r for r in _rows(tmp_path) if r.get("kind") == "step"]
    assert len(steps) == 1
    assert steps[0]["step"] == 10


def test_writer_never_raises_into_training(tmp_path):
    writer = JobEventWriter(tmp_path)
    writer.path.chmod(0o444)  # read-only file -> open fails on POSIX
    writer.write({"kind": "phase", "phase": "training"})
    writer.path.chmod(0o644)


def test_stop_signal_single_saving_phase_and_last_step(tmp_path):
    """Fix-round #3/#12c: ``on_step_end`` fires on EVERY step while
    control.json persists, so the ``saving`` phase must emit once — and the
    callback tracks the highest step seen (cli.py's done-event steps_done)."""
    (tmp_path / "control.json").write_text('{"action": "stop_save"}')
    cb = UiFileEventCallback(tmp_path)
    args, control = _Args(), _Control()
    cb.on_train_begin(args, _State(), control)
    for step in (1, 2, 3):
        control = cb.on_step_end(args, _State(step=step), control)
        assert control.should_training_stop
        assert control.should_save
    savings = (
        r for r in _rows(tmp_path)
        if r.get("kind") == "phase" and r.get("phase") == "saving"
    )
    assert len(list(savings)) == 1
    assert cb.last_step == 3
