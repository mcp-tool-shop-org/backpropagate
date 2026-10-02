# ui-v2 P2: Runs + Run detail for runs started from the UI.
"""UI jobs keep their history in ``jobs/<job>/output/run_history.json``.

These pin what the Runs table and the run page show for them: the job's own
outcome (a cooperative stop is "stopped", not "completed"), readable start
times, the dataset's file name (no home directory), and on the run page the
loss curve from the job's progress events and its output.log.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from backpropagate import ui_state as us  # noqa: E402
from backpropagate.checkpoints import RunHistoryManager  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    out = (home / ".backpropagate" / "ui-outputs").resolve()
    out.mkdir(parents=True)
    return SimpleNamespace(home=home.resolve(), out=out)


def _ui_job(sandbox, job="run_20261002_000000_1_abcd", *, run_id="uijob0001", status="stopped"):
    """A finished UI job: history entry, job.json, events with loss, output.log."""
    job_dir = sandbox.out / "jobs" / job
    output = job_dir / "output"
    output.mkdir(parents=True)
    RunHistoryManager(str(output)).record_run({
        "run_id": run_id,
        "status": "completed",
        "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dataset_info": str(sandbox.home / "data" / "p2.jsonl"),
        "started_at": "2026-10-02T04:45:42.269557",
        "duration_seconds": 25.0,
        "final_loss": 0.755,
    })
    (job_dir / "job.json").write_text(json.dumps({"status": status}))
    rows = [{"kind": "phase", "phase": "training"}]
    rows += [{"kind": "step", "step": s, "loss": 3.0 / s} for s in (10, 20, 30)]
    (job_dir / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (job_dir / "output.log").write_text(
        f"==> Loading model from {sandbox.home}\n 10%|#  | 1/10\r 20%|## | 2/10\nTraining complete!\n"
    )
    return job_dir


def _detail(rid):
    import reflex as rx
    from reflex.istate.data import RouterData

    root = rx.State(_reflex_internal_init=True)
    root.router = RouterData.from_router_data(
        {"pathname": f"/runs/{rid}", "query": {"rid": rid}, "headers": {}, "ip": "127.0.0.1"}
    )
    return root.substates[us.RunDetailState.get_name()]


def test_runs_table_shows_a_stopped_ui_job_as_stopped(sandbox):
    _ui_job(sandbox)
    s = us.RunsState()
    s.load_runs()
    row = s.runs[0]
    assert row["status"] == "stopped"
    assert row["started_at"] == "2026-10-02 04:45"
    assert row["dataset"] == "p2.jsonl"


def test_runs_table_keeps_completed_for_a_job_that_finished(sandbox):
    _ui_job(sandbox, status="done")
    s = us.RunsState()
    s.load_runs()
    assert s.runs[0]["status"] == "completed"


def test_run_detail_finds_a_ui_job_and_reads_its_files(sandbox):
    _ui_job(sandbox)
    d = _detail("uijob0001")
    d.load_run()
    assert d.not_found is False
    assert d.status == "stopped"
    assert d.started_at == "2026-10-02 04:45"
    assert d.dataset == "p2.jsonl"
    assert d.loss_history == [0.3, 0.15, 0.1]
    # output.log tail: tqdm frames collapsed, home directory redacted.
    assert "Training complete!" in d.log_lines
    assert " 20%|## | 2/10" in d.log_lines
    assert not any(str(sandbox.home) in line for line in d.log_lines)


def test_run_detail_actions_resolve_the_job_history(sandbox):
    _ui_job(sandbox)
    d = _detail("uijob0001")
    d.load_run()
    d.replay()
    assert d.action_error == "" and "replayable" in d.action_result
    d.delete_run()
    assert d.was_deleted is True


@pytest.mark.parametrize(
    ("value", "label"),
    [
        ("/home/alice/data/train.jsonl", "train.jsonl"),
        ("C:\\Users\\alice\\data\\train.jsonl", "train.jsonl"),
        ("org/dataset-name", "org/dataset-name"),
        ("train.jsonl", "train.jsonl"),
        ("", "-"),
        (None, "-"),
    ],
)
def test_dataset_label(value, label):
    assert us._dataset_label(value) == label


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        ("2026-10-02T04:45:42.269557", "2026-10-02 04:45"),
        ("2026-10-02T08:45:42Z", "2026-10-02 08:45"),
        ("not a date at all", "not a date at all"[:16]),
        (None, "-"),
    ],
)
def test_fmt_started(value, shown):
    assert us._fmt_started(value) == shown


def test_export_the_model_hands_the_job_output_to_the_export_page(sandbox):
    job_dir = _ui_job(sandbox)
    d = _detail("uijob0001")
    d.load_run()
    assert d.can_export_model is True
    events = d.export_model()
    names = [getattr(getattr(e, "handler", None), "fn", None) for e in events]
    assert any(fn is not None and fn.__name__ == "set_source_model_path" for fn in names)
    sent = str(events[0].args[0][1])  # a JS string literal (backslashes escaped)
    assert job_dir.name in sent and sent.rstrip('"').endswith("output")


# ---- Models page: the cache the machine really uses -----------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"HF_HUB_CACHE": "D:/hub-cache"}, "D:/hub-cache"),
        ({"HF_HOME": "E:/AI-Models/hf-cache"}, "E:/AI-Models/hf-cache/hub"),
        ({"XDG_CACHE_HOME": "/xdg"}, "/xdg/huggingface/hub"),
        ({}, None),  # ~/.cache/huggingface/hub
    ],
)
def test_hf_hub_cache_dir_follows_the_hub_env_order(sandbox, monkeypatch, env, expected):
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    got = us._hf_hub_cache_dir()
    if expected is None:
        assert got == sandbox.home / ".cache" / "huggingface" / "hub"
    else:
        assert got == Path(expected)


def test_home_relative_hides_the_username(sandbox):
    assert us._home_relative(str(sandbox.home / ".cache" / "hf")) == "~/.cache/hf"
    outside = str(Path(sandbox.home).parent.parent / "elsewhere")
    assert us._home_relative(outside) == outside
