# Runs page storage line and clean-up (ui_jobs.JobManager.storage_summary / clean_up).
"""Nothing is removed automatically, and the clean-up never removes a model.

A job folder is removable only when it holds no saved model, is not the job
that is running, and is a real ``run_*`` directory directly inside the jobs
folder (never a link, never anything else).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from backpropagate import ui_jobs
from backpropagate.ui_jobs import JobHandle, JobManager, JobSpec


def _job(root: Path, name: str, files: dict[str, int]) -> Path:
    folder = root / name
    for rel, size in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    return folder


@pytest.fixture
def jobs(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    _job(root, "run_failed", {"events.jsonl": 100, "output.log": 400})
    _job(root, "run_measure", {"events.jsonl": 50})
    _job(root, "run_trained", {"events.jsonl": 10, "output/adapter_model.safetensors": 5000})
    return root


def _link(target: Path, link: Path) -> None:
    """A directory link: a junction on Windows (no privilege needed), a symlink elsewhere."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


class _Proc:
    pid = 99

    def __init__(self, code):
        self._code = code

    def poll(self):
        return self._code


def _track(manager: JobManager, folder: Path, proc: _Proc) -> None:
    spec = JobSpec(kind="sft", model="org/tiny", dataset_path="d.jsonl")
    manager._handles[folder.name] = JobHandle(
        job_id=folder.name, run_dir=folder, pid=99, started_at=0.0, spec=spec
    )
    manager._jobs[folder.name] = proc


def test_summary_counts_every_folder_and_what_a_clean_up_would_free(jobs):
    out = JobManager(jobs_root=jobs).storage_summary()
    assert out == {
        "folders": 3, "bytes": 5560, "removable_folders": 2, "removable_bytes": 550,
    }
    assert sorted(p.name for p in jobs.iterdir()) == ["run_failed", "run_measure", "run_trained"]


def test_clean_up_removes_only_folders_without_a_model(jobs):
    out = JobManager(jobs_root=jobs).clean_up()
    assert out == {"removed": 2, "bytes": 550, "errors": 0}
    assert [p.name for p in jobs.iterdir()] == ["run_trained"]
    assert (jobs / "run_trained" / "output" / "adapter_model.safetensors").stat().st_size == 5000
    # Nothing left to remove.
    assert JobManager(jobs_root=jobs).clean_up() == {"removed": 0, "bytes": 0, "errors": 0}


@pytest.mark.parametrize(
    "model_file",
    [
        "output/model.safetensors",
        "output/checkpoint-50/adapter_model.safetensors",
        "output/model-Q4_K_M.GGUF",
        "output/adapter_config.json",
        "output/adapter_model.bin",
        "output/pytorch_model.bin",
    ],
)
def test_any_saved_model_file_protects_the_folder(tmp_path, model_file):
    root = tmp_path / "jobs"
    _job(root, "run_a", {"events.jsonl": 10, model_file: 20})
    manager = JobManager(jobs_root=root)
    assert manager.storage_summary()["removable_folders"] == 0
    assert manager.clean_up()["removed"] == 0
    assert (root / "run_a" / model_file).exists()


def test_the_running_job_is_never_removed(jobs):
    manager = JobManager(jobs_root=jobs)
    _track(manager, jobs / "run_failed", _Proc(None))
    assert manager.storage_summary()["removable_folders"] == 1
    assert manager.clean_up() == {"removed": 1, "bytes": 50, "errors": 0}
    assert (jobs / "run_failed" / "output.log").exists()
    assert "run_failed" in manager._handles


def test_a_finished_job_is_forgotten_when_its_folder_is_removed(jobs):
    manager = JobManager(jobs_root=jobs)
    _track(manager, jobs / "run_failed", _Proc(1))
    assert manager.clean_up()["removed"] == 2
    assert "run_failed" not in manager._handles
    assert "run_failed" not in manager._jobs


def test_a_link_is_neither_counted_nor_followed(jobs, tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "notes.txt").write_bytes(b"y" * 900)
    try:
        _link(outside, jobs / "run_link")
    except OSError as exc:  # pragma: no cover - links unavailable on this machine
        pytest.skip(f"cannot create a directory link here: {exc}")
    manager = JobManager(jobs_root=jobs)
    assert manager.storage_summary()["folders"] == 3
    assert manager.clean_up()["removed"] == 2
    assert (outside / "notes.txt").stat().st_size == 900
    assert (jobs / "run_link").exists()


def test_a_link_inside_a_job_folder_is_not_followed(jobs, tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(b"y" * 900)
    try:
        _link(outside, jobs / "run_failed" / "linked")
    except OSError as exc:  # pragma: no cover - links unavailable on this machine
        pytest.skip(f"cannot create a directory link here: {exc}")
    manager = JobManager(jobs_root=jobs)
    # The linked model belongs to another folder, and its bytes are not counted.
    assert manager.storage_summary() == {
        "folders": 3, "bytes": 5560, "removable_folders": 2, "removable_bytes": 550,
    }
    assert manager.clean_up()["removed"] == 2
    assert (outside / "model.safetensors").stat().st_size == 900


def test_only_run_folders_directly_inside_the_jobs_folder(jobs):
    _job(jobs, "notes", {"readme.txt": 70})
    (jobs / "run_file").write_bytes(b"z" * 30)
    manager = JobManager(jobs_root=jobs)
    assert manager.storage_summary()["folders"] == 3
    manager.clean_up()
    assert (jobs / "notes" / "readme.txt").exists()
    assert (jobs / "run_file").exists()


def test_a_folder_that_cannot_be_removed_is_reported_not_raised(jobs, monkeypatch):
    real = shutil.rmtree

    def rmtree(path, *a, **k):
        if Path(path).name == "run_failed":
            raise PermissionError("in use")
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    out = JobManager(jobs_root=jobs).clean_up()
    assert out == {"removed": 1, "bytes": 50, "errors": 1}
    assert (jobs / "run_failed").exists()


def test_missing_jobs_folder_is_empty_not_an_error(tmp_path):
    manager = JobManager(jobs_root=tmp_path / "nope")
    assert manager.storage_summary()["folders"] == 0
    assert manager.clean_up() == {"removed": 0, "bytes": 0, "errors": 0}


def test_default_jobs_folder_outside_the_sandbox_is_left_alone(jobs, tmp_path, monkeypatch):
    import backpropagate.ui_security as sec

    sandbox = tmp_path / "ui-outputs"
    sandbox.mkdir()
    monkeypatch.setattr(sec, "get_ui_output_dir", lambda: sandbox)
    manager = JobManager()
    manager.jobs_root = jobs  # where a link planted at <sandbox>/jobs would resolve
    assert manager.storage_summary()["folders"] == 0
    assert manager.clean_up()["removed"] == 0
    assert (jobs / "run_failed").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX: a job can outlive the UI that started it")
def test_a_job_left_running_by_an_earlier_session_is_kept(jobs, monkeypatch):
    (jobs / "run_failed" / ui_jobs.JOB_FILENAME).write_text(
        json.dumps({"job_id": "run_failed", "status": "running", "pid": 4242}), encoding="utf-8"
    )
    monkeypatch.setattr(ui_jobs, "_posix_process_is_ours", lambda pid: pid == 4242)
    manager = JobManager(jobs_root=jobs)
    assert manager.storage_summary()["removable_folders"] == 1
    assert manager.clean_up()["removed"] == 1
    assert (jobs / "run_failed").exists()
