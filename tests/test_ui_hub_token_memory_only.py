# The Export page's HuggingFace token never reaches disk (external review 2026-10-02, C-1).
"""Reflex pickles every var of a state, backend vars included, to
``<workdir>/.states/*.pkl`` and keeps the files after the server stops. The
token typed into the Export page used to be a backend var, so it was written
there in the clear. It now lives in process memory only, keyed by the browser
session, and each launch removes saved session state left by an earlier run.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from backpropagate import ui_state as us  # noqa: E402
from backpropagate import ui_workdir  # noqa: E402

SECRET = "hf_" + "Zq9" * 12


def _state(client_token: str = "") -> us.ExportState:
    import reflex as rx
    from reflex.istate.data import RouterData

    root = rx.State(_reflex_internal_init=True)
    root.router = RouterData.from_router_data(
        {"pathname": "/export", "query": {}, "headers": {}, "ip": "127.0.0.1",
         "token": client_token}
    )
    return root.substates[us.ExportState.get_name()]


def test_the_token_is_not_in_what_reflex_writes_to_disk():
    s = _state("tab-1")
    s.set_hub_token(SECRET)
    assert s.hub_token_set is True
    assert s._hub_token_value() == SECRET
    assert SECRET.encode() not in s._serialize()
    assert SECRET.encode() not in s.parent_state._serialize()


def test_each_browser_session_has_its_own_token():
    one, two = _state("tab-1"), _state("tab-2")
    one.set_hub_token(SECRET)
    assert two._hub_token_value() == ""
    two.set_hub_token("hf_" + "b" * 30)
    assert one._hub_token_value() == SECRET
    one.set_hub_token("")
    assert one._hub_token_value() == ""
    assert two._hub_token_value() == "hf_" + "b" * 30


def test_abandoned_sessions_do_not_pile_up():
    for i in range(us._HUB_TOKENS_MAX + 5):
        us._hub_token_put(f"tab-{i}", f"hf_{i:030d}")
    assert len(us._HUB_TOKENS) == us._HUB_TOKENS_MAX
    assert us._hub_token_get("tab-0") == ""  # oldest dropped first
    assert us._hub_token_get(f"tab-{us._HUB_TOKENS_MAX + 4}")


def test_setting_again_moves_a_session_to_the_newest_slot():
    us._hub_token_put("keep", "hf_" + "k" * 30)
    for i in range(us._HUB_TOKENS_MAX - 1):
        us._hub_token_put(f"tab-{i}", f"hf_{i:030d}")
    us._hub_token_put("keep", "hf_" + "k" * 30)
    us._hub_token_put("one-more", "hf_" + "m" * 30)
    assert us._hub_token_get("keep")
    assert us._hub_token_get("tab-0") == ""


def test_a_push_after_a_restart_asks_for_the_token_again(monkeypatch):
    import backpropagate.export as export_mod

    called = []
    monkeypatch.setattr(export_mod, "push_to_hub", lambda **kw: called.append(kw), raising=False)
    s = _state("tab-1")
    s.source_model_path = "model-out"
    s.set_hub_repo_id("owner/repo")
    s.set_hub_token(SECRET)
    us._HUB_TOKENS.clear()  # what a restart does: the state file survives, the memory does not
    s.push_to_hub()
    assert called == []
    assert s.hub_status == "error"
    assert "again" in s.hub_message
    assert s.hub_token_set is False


def test_a_successful_push_forgets_the_token(monkeypatch, tmp_path):
    from pathlib import Path

    import backpropagate.export as export_mod

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    adapter = home / ".backpropagate" / "ui-outputs" / "adapter"
    adapter.mkdir(parents=True)

    seen = {}
    monkeypatch.setattr(export_mod, "push_to_hub", lambda **kw: seen.update(kw), raising=False)
    s = _state("tab-1")
    s.set_source_model_path(str(adapter))
    s.set_hub_repo_id("owner/repo")
    s.set_hub_token(SECRET)
    s.push_to_hub()
    assert seen["token"] == SECRET
    assert s.hub_status == "done"
    assert us._HUB_TOKENS == {}
    assert s.hub_token_set is False


# ---- saved session state from an earlier run is removed at launch ---------------------


def _package_dir(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "rxconfig.py").write_text('app_module_import="ui_app.app"\n', encoding="utf-8")
    return package


def test_launch_removes_saved_session_state(tmp_path, monkeypatch):
    workdir = tmp_path / "wd"
    states = workdir / ".states"
    states.mkdir(parents=True)
    (states / "abc.pkl").write_bytes(b"old state with " + SECRET.encode())
    (workdir / ".web").mkdir()
    (workdir / ".web" / "keep.txt").write_text("build output", encoding="utf-8")
    monkeypatch.setenv(ui_workdir.WORKDIR_ENV_VAR, str(workdir))
    ui_workdir.ensure_ui_workdir(_package_dir(tmp_path), version="1.8.2")
    assert not states.exists()
    assert (workdir / ".web" / "keep.txt").exists()


def test_launch_removes_saved_session_state_in_the_default_location(tmp_path, monkeypatch):
    monkeypatch.delenv(ui_workdir.WORKDIR_ENV_VAR, raising=False)
    monkeypatch.setattr(ui_workdir, "default_ui_work_root", lambda: tmp_path / "root")
    package = _package_dir(tmp_path)
    first = ui_workdir.ensure_ui_workdir(package, version="1.8.2")
    (first / ".states").mkdir()
    (first / ".states" / "abc.pkl").write_bytes(SECRET.encode())
    again = ui_workdir.ensure_ui_workdir(package, version="1.8.2")
    assert again == first
    assert not (first / ".states").exists()


def test_a_linked_states_folder_is_left_alone(tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "precious.txt").write_text("keep", encoding="utf-8")
    try:
        os.symlink(elsewhere, workdir / ".states", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")
    warnings: list[str] = []
    ui_workdir.clear_saved_ui_state(workdir, warnings.append)
    assert (elsewhere / "precious.txt").exists()
    assert warnings == []


def test_a_states_folder_that_cannot_be_removed_warns(tmp_path, monkeypatch):
    workdir = tmp_path / "wd"
    (workdir / ".states").mkdir(parents=True)

    def refuse(path, *a, **k):
        raise PermissionError("in use")

    monkeypatch.setattr(ui_workdir.shutil, "rmtree", refuse)
    warnings: list[str] = []
    ui_workdir.clear_saved_ui_state(workdir, warnings.append)
    assert len(warnings) == 1 and "saved UI session state" in warnings[0]
