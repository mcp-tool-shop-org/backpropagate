# ui-v2 P2: multi-run + export through the UI job runner.
"""Job-runner, CLI-wrapper and callback tests for the P2 job kinds.

Fake processes and fake command bodies; no GPU, no model downloads.
"""

from __future__ import annotations

import argparse
import json

import pytest

from backpropagate.job_events import (
    CONTROL_FILENAME,
    EVENTS_FILENAME,
    JobEventWriter,
    UiFileEventCallback,
)
from backpropagate.ui_jobs import (
    UI_GGUF_QUANTS,
    JobManager,
    JobSpec,
    JobValidationError,
)


class _FakeProc:
    pid = 4242
    returncode = None
    _handle = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    import backpropagate.ui_security as sec

    monkeypatch.setattr(sec, "get_ui_output_dir", lambda: tmp_path)
    return tmp_path


def _manager(tmp_path):
    captured: dict = {}

    def spawn(argv, **kw):
        captured["argv"] = argv
        return _FakeProc()

    return JobManager(jobs_root=tmp_path / "jobs", spawn=spawn), captured


def _data(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "hi"}\n')
    return str(data)


def _adapter(tmp_path):
    out = tmp_path / "jobs" / "run_old" / "output"
    out.mkdir(parents=True)
    (out / "adapter_config.json").write_text("{}")
    return str(out)


def _flag(argv, name):
    return argv[argv.index(name) + 1]


# ---- argv per kind -----------------------------------------------------------


@pytest.mark.parametrize(
    ("merge", "expected_mode", "expected_strategy"),
    [("slao", "slao", None), ("simple", "simple", None), ("ties", "slao", "ties")],
)
def test_multi_run_argv_maps_merge_choice_to_cli_flags(
    tmp_path, merge, expected_mode, expected_strategy
):
    m, captured = _manager(tmp_path)
    job = m.start(
        JobSpec(
            kind="multi_run",
            model="org/tiny",
            dataset_path=_data(tmp_path),
            runs=2,
            steps=15,
            samples=40,
            merge=merge,
        )
    )
    argv = captured["argv"]
    assert argv[3] == "multi-run"
    assert _flag(argv, "--runs") == "2"
    assert _flag(argv, "--steps") == "15"
    assert _flag(argv, "--samples") == "40"
    assert _flag(argv, "--merge-mode") == expected_mode
    if expected_strategy:
        assert _flag(argv, "--merge-strategy") == expected_strategy
    else:
        assert "--merge-strategy" not in argv
    assert _flag(argv, "--ui-run-dir") == str(job.run_dir)
    # Flags the multi-run CLI does not have must never be sent.
    for absent in ("--lr", "--lora-r", "--batch-size"):
        assert absent not in argv


def test_export_gguf_argv_with_ollama(tmp_path):
    m, captured = _manager(tmp_path)
    source = _adapter(tmp_path)
    job = m.start(
        JobSpec(
            kind="export",
            source_path=source,
            export_format="gguf",
            quantization="q4_k_m",
            ollama_name="my-model",
        )
    )
    argv = captured["argv"]
    assert argv[3:5] == ["export", source]
    assert _flag(argv, "--format") == "gguf"
    assert _flag(argv, "--quantization") == "q4_k_m"
    assert _flag(argv, "--ollama-name") == "my-model"
    assert "--ollama" in argv
    assert _flag(argv, "--output") == str(job.run_dir / "output")


def test_export_lora_argv_has_no_gguf_flags(tmp_path):
    m, captured = _manager(tmp_path)
    m.start(JobSpec(kind="export", source_path=_adapter(tmp_path), export_format="lora"))
    argv = captured["argv"]
    assert "--quantization" not in argv
    assert "--ollama" not in argv


# ---- validation ----------------------------------------------------------------


def test_ui_gguf_levels_match_the_cli_choices():
    from backpropagate.cli import create_parser

    parser = create_parser()
    sub = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ).choices["export"]
    quant = next(a for a in sub._actions if "--quantization" in a.option_strings)
    assert tuple(quant.choices) == UI_GGUF_QUANTS


@pytest.mark.parametrize(
    "over",
    [
        {"export_format": "safetensors"},
        {"export_format": "gguf", "quantization": "q3_k_m"},
        {"export_format": "lora", "ollama_name": "x"},
        {"export_format": "gguf", "ollama_name": "../evil"},
        {"source_path": ""},
    ],
)
def test_export_validation_rejects(tmp_path, over):
    m, _ = _manager(tmp_path)
    spec = JobSpec(kind="export", source_path=_adapter(tmp_path))
    for k, v in over.items():
        setattr(spec, k, v)
    with pytest.raises(JobValidationError):
        m.start(spec)


def test_export_source_must_be_inside_the_sandbox(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere")
    (outside / "adapter_config.json").write_text("{}")
    m, _ = _manager(tmp_path)
    with pytest.raises(JobValidationError, match="only reads files inside"):
        m.start(JobSpec(kind="export", source_path=str(outside)))


@pytest.mark.parametrize("over", [{"runs": 0}, {"runs": 51}, {"merge": "weighted"}])
def test_multi_run_validation_rejects(tmp_path, over):
    m, _ = _manager(tmp_path)
    spec = JobSpec(kind="multi_run", model="org/tiny", dataset_path=_data(tmp_path))
    for k, v in over.items():
        setattr(spec, k, v)
    with pytest.raises(JobValidationError):
        m.start(spec)


# ---- the CLI wrapper (_run_as_ui_job) ------------------------------------------


def _rows(run_dir):
    return [
        json.loads(x)
        for x in (run_dir / EVENTS_FILENAME).read_text().splitlines()
        if x.strip()
    ]


def _args(run_dir):
    return argparse.Namespace(ui_run_dir=str(run_dir))


def test_wrapper_without_ui_dir_just_runs_the_body(tmp_path):
    from backpropagate.cli import _run_as_ui_job

    assert _run_as_ui_job(argparse.Namespace(), lambda a: 7, "exporting") == 7


def test_wrapper_records_done_with_output_path(tmp_path):
    from backpropagate.cli import EXIT_OK, _run_as_ui_job

    def body(args):
        args._ui_job.output_path = "/out/model.gguf"
        return EXIT_OK

    assert _run_as_ui_job(_args(tmp_path), body, "exporting") == EXIT_OK
    rows = _rows(tmp_path)
    assert [r["phase"] for r in rows if r["kind"] == "phase"] == ["exporting", "done"]
    done = [r for r in rows if r["kind"] == "done"][-1]
    assert done["status"] == "done"
    assert done["output_path"] == "/out/model.gguf"
    assert json.loads((tmp_path / "job.json").read_text())["status"] == "done"


def test_wrapper_reports_stopped_when_control_file_present(tmp_path):
    from backpropagate.cli import EXIT_OK, _run_as_ui_job

    (tmp_path / CONTROL_FILENAME).write_text('{"action": "stop_save"}')
    _run_as_ui_job(_args(tmp_path), lambda a: EXIT_OK, "loading")
    assert [r for r in _rows(tmp_path) if r["kind"] == "done"][-1]["status"] == "stopped"


def test_wrapper_failure_reads_the_structured_code_from_the_log(tmp_path):
    from backpropagate.cli import EXIT_RUNTIME_ERROR, _run_as_ui_job

    (tmp_path / "output.log").write_text(
        "==> Loading model\n"
        "[ERROR] [DEP_MODEL_LOAD_FAILED] Could not load org/x\n"
        "[INFO] Suggestion: check the name\n"
    )
    rc = _run_as_ui_job(_args(tmp_path), lambda a: EXIT_RUNTIME_ERROR, "exporting")
    assert rc == EXIT_RUNTIME_ERROR
    err = [r for r in _rows(tmp_path) if r["kind"] == "error"][-1]
    assert err["code"] == "DEP_MODEL_LOAD_FAILED"
    assert "Could not load" in err["message"]
    job = json.loads((tmp_path / "job.json").read_text())
    assert job["status"] == "failed" and job["error_code"] == "DEP_MODEL_LOAD_FAILED"


def test_wrapper_failure_without_a_code_falls_back_by_exit_code(tmp_path):
    from backpropagate.cli import EXIT_USER_ERROR, _run_as_ui_job

    _run_as_ui_job(_args(tmp_path), lambda a: EXIT_USER_ERROR, "exporting")
    assert [r for r in _rows(tmp_path) if r["kind"] == "error"][-1]["code"] == "INPUT_INVALID"


def test_wrapper_records_an_exception_and_reraises(tmp_path):
    from backpropagate.cli import _run_as_ui_job

    def body(args):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _run_as_ui_job(_args(tmp_path), body, "loading")
    assert [r for r in _rows(tmp_path) if r["kind"] == "error"][-1]["message"] == "boom"


# ---- multi-run progress callback -------------------------------------------------


class _A:
    logging_steps = 1


class _S:
    def __init__(self, step, max_steps=10):
        self.global_step = step
        self.max_steps = max_steps
        self.epoch = 1.0


class _C:
    should_training_stop = False
    should_save = False


def test_multi_run_steps_are_session_wide(tmp_path):
    writer = JobEventWriter(tmp_path)
    cb = UiFileEventCallback(tmp_path, writer=writer, total_steps=30)
    cb.on_train_begin(_A(), _S(0), _C())
    cb.on_log(_A(), _S(5), _C(), logs={"loss": 1.0})
    cb.step_offset = 10  # run 2 of 3, 10 steps per run
    cb.on_train_begin(_A(), _S(0), _C())
    cb.on_log(_A(), _S(5), _C(), logs={"loss": 0.8})
    cb.on_step_end(_A(), _S(5), _C())
    steps = [r for r in _rows(tmp_path) if r["kind"] == "step"]
    assert [(r["step"], r["total_steps"]) for r in steps] == [(5, 30), (15, 30)]
    assert cb.last_step == 15


def test_stop_calls_the_on_stop_hook_once(tmp_path):
    calls = []
    cb = UiFileEventCallback(tmp_path, on_stop=calls.append)
    (tmp_path / CONTROL_FILENAME).write_text('{"action": "stop_save"}')
    for step in (1, 2, 3):
        control = cb.on_step_end(_A(), _S(step), _C())
        assert control.should_training_stop and control.should_save
    assert calls == ["Stopped from the web UI"]


def test_run_marker_event(tmp_path):
    JobEventWriter(tmp_path).run_marker(2, 3)
    assert _rows(tmp_path)[-1] == {**_rows(tmp_path)[-1], "kind": "run", "run": 2, "runs": 3}


def test_multi_run_trainer_installs_extra_callbacks():
    from backpropagate.multi_run import MultiRunTrainer

    marker = object()
    trainer = MultiRunTrainer(model="org/tiny", extra_callbacks=[marker])
    assert trainer._extra_callbacks == [marker]
