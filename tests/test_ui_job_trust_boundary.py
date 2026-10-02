# The browser cannot choose where a UI job writes.
"""``TrainState.start_job`` is an event handler, so its payload comes from the
browser. Before this fix it copied every JobSpec field from the payload,
including ``output_dir`` and ``scratch_root``: an authenticated client (the
``--share`` / ``--auth`` case) could make a training or export job write
outside the UI output sandbox. These pin both layers of the fix: the handler
drops the server-only fields, and the JobManager refuses a spec whose write
locations leave the sandbox.
"""

from __future__ import annotations

import pytest

from backpropagate import ui_jobs
from backpropagate.ui_jobs import SERVER_ONLY_SPEC_FIELDS, JobSpec, JobValidationError


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    import backpropagate.ui_security as sec

    box = tmp_path / "ui-outputs"
    box.mkdir()
    monkeypatch.setattr(sec, "get_ui_output_dir", lambda: box)
    data = box / "d.jsonl"
    data.write_text('{"text": "hi"}\n', encoding="utf-8")
    return box, data, tmp_path / "outside"


def _spec(data, **over):
    return JobSpec(kind="sft", model="org/tiny", dataset_path=str(data), **over)


def test_output_dir_outside_the_sandbox_is_refused(sandbox):
    box, data, outside = sandbox
    with pytest.raises(JobValidationError, match="only writes inside"):
        ui_jobs._validate_spec(_spec(data, output_dir=str(outside)))
    with pytest.raises(JobValidationError, match="only writes inside"):
        ui_jobs._validate_spec(_spec(data, output_dir=str(box / ".." / "outside")))
    ui_jobs._validate_spec(_spec(data, output_dir=str(box / "runs" / "new")))  # not created yet: fine


def test_scratch_root_outside_the_sandbox_is_refused(sandbox):
    box, data, outside = sandbox
    with pytest.raises(JobValidationError, match="only writes inside"):
        ui_jobs._validate_spec(_spec(data, scratch_root=str(outside)))
    ui_jobs._validate_spec(_spec(data, scratch_root=str(box)))


def test_export_output_dir_is_checked_too(sandbox):
    box, _data, outside = sandbox
    adapter = box / "adapter"
    adapter.mkdir()
    spec = JobSpec(kind="export", source_path=str(adapter), export_format="lora",
                   output_dir=str(outside))
    with pytest.raises(JobValidationError, match="only writes inside"):
        ui_jobs._validate_spec(spec)


def test_server_only_fields_are_named():
    assert {"scratch_root", "output_dir", "trust_remote_code"} == SERVER_ONLY_SPEC_FIELDS
    assert set(JobSpec.__dataclass_fields__) >= SERVER_ONLY_SPEC_FIELDS


def test_start_job_drops_server_only_fields_from_the_payload(sandbox, monkeypatch):
    pytest.importorskip("reflex")
    from backpropagate import ui_state as us

    _box, data, outside = sandbox
    seen = {}
    state = us.TrainState()
    monkeypatch.setattr(
        us.TrainState, "_begin_job", lambda self, spec: seen.setdefault("spec", spec)
    )
    state.start_job({
        "kind": "multi_run", "model": "org/tiny", "dataset_path": str(data), "runs": 2,
        "output_dir": str(outside), "scratch_root": str(outside), "trust_remote_code": True,
        "not_a_field": 1,
    })
    spec = seen["spec"]
    assert spec.kind == "multi_run" and spec.runs == 2
    assert spec.output_dir is None and spec.scratch_root is None
    assert spec.trust_remote_code is False
