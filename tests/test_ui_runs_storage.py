# Runs page: the storage line and its Clean up action (RunsState).
"""What the Runs page says about disk use, and what Clean up reports.

The removal rules themselves are pinned in ``test_ui_jobs_storage.py``; these
cover the page state on top: the labels, the count that enables the button,
and the result line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from backpropagate import ui_jobs  # noqa: E402
from backpropagate import ui_state as us  # noqa: E402
from backpropagate.ui_jobs import JobManager  # noqa: E402


def _job(root: Path, name: str, files: dict[str, int]) -> None:
    for rel, size in files.items():
        path = root / name / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    mgr = JobManager(jobs_root=root)
    monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: mgr)
    return mgr


def test_storage_line_with_something_to_clean_up(manager):
    _job(manager.jobs_root, "run_failed", {"output.log": 3 * 1024})
    _job(manager.jobs_root, "run_trained", {"output/adapter_model.safetensors": 2 * 1024 * 1024})
    s = us.RunsState()
    s._load_storage()
    assert s.storage_label == "2 job folders · 2.0 MB"
    assert s.storage_removable_label == "1 without a saved model · 3 KB"
    assert s.storage_removable_count == 1


def test_storage_line_when_every_folder_holds_a_model(manager):
    _job(manager.jobs_root, "run_trained", {"output/adapter_model.safetensors": 2048})
    s = us.RunsState()
    s._load_storage()
    assert s.storage_label == "1 job folder · 2 KB"
    assert s.storage_removable_label == ""
    assert s.storage_removable_count == 0


def test_no_job_folders_hides_the_line(manager):
    s = us.RunsState()
    s._load_storage()
    assert (s.storage_label, s.storage_removable_label, s.storage_removable_count) == ("", "", 0)


def test_a_failing_summary_hides_the_line_instead_of_raising(manager, monkeypatch):
    def boom():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(manager, "storage_summary", boom)
    s = us.RunsState()
    s.storage_label = "stale"
    s.storage_removable_count = 4
    s._load_storage()
    assert (s.storage_label, s.storage_removable_label, s.storage_removable_count) == ("", "", 0)


def test_clean_up_reports_what_was_freed_and_reloads(manager):
    _job(manager.jobs_root, "run_failed", {"output.log": 3 * 1024})
    _job(manager.jobs_root, "run_trained", {"output/adapter_model.safetensors": 2048})
    s = us.RunsState()
    follow_up = s.clean_up_storage()
    assert s.storage_result == "Removed 1 folder, freed 3 KB."
    assert follow_up is us.RunsState.load_runs
    assert not (manager.jobs_root / "run_failed").exists()
    assert (manager.jobs_root / "run_trained").exists()


def test_clean_up_mentions_folders_it_could_not_remove(manager, monkeypatch):
    monkeypatch.setattr(
        manager, "clean_up", lambda: {"removed": 2, "bytes": 5 * 1024 * 1024, "errors": 1}
    )
    s = us.RunsState()
    s.clean_up_storage()
    assert s.storage_result == "Removed 2 folders, freed 5.0 MB. 1 could not be removed (in use?)."


def test_clean_up_failure_is_shown_with_the_home_directory_redacted(manager, monkeypatch):
    home = str(Path.home())

    def boom():
        raise OSError(f"cannot read {home}\\jobs")

    monkeypatch.setattr(manager, "clean_up", boom)
    s = us.RunsState()
    assert s.clean_up_storage() is None
    assert s.storage_result.startswith("Clean-up failed:")
    assert home not in s.storage_result
