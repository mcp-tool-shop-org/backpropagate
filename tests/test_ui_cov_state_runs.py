"""Behaviour tests for the data-side Reflex states: Runs, RunDetail, Models, Dataset upload.

These states read and write the filesystem (run history, HF cache, uploads),
so every test runs against a real directory tree under ``tmp_path`` with
``Path.home`` pointed at it - ``RunHistoryManager`` is the real store, not a
fake. The assertions are on state fields, the on-disk effects, and the
operator-facing strings (which must never carry the home dir / username).

Mocked, and why: ``subprocess.run`` / ``shutil.which`` for the one handler that
shells out to the ``backprop`` CLI (``diff_against``) so the exact argv can be
asserted without a CLI on PATH; the router object (Reflex supplies it at
runtime); and individual filesystem calls (``Path.iterdir`` / ``open`` /
``Path.unlink`` / ``shutil.rmtree``) where a test needs an ``OSError`` that a
healthy tmp dir cannot produce.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import subprocess
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
    # The Models page follows HF_HOME & co. (ui-v2 P2); keep the cache under
    # the fake home whatever the machine running the tests has set.
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)
    return SimpleNamespace(
        home=home.resolve(),
        out=(home / ".backpropagate" / "ui-outputs").resolve(),
        cache=(home / ".cache" / "huggingface" / "hub").resolve(),
    )


def _seed(sandbox, **entry):
    """Record one run in the sandbox history via the real manager."""
    sandbox.out.mkdir(parents=True, exist_ok=True)
    mgr = RunHistoryManager(str(sandbox.out))
    base = {"run_id": "run00001", "status": "completed", "model_name": "Qwen/Qwen2.5-7B",
            "dataset_info": "data.jsonl", "started_at": "2026-09-01T10:00:00",
            "duration_seconds": 12.4, "final_loss": 0.123456}
    base.update(entry)
    return mgr.record_run(base)


def _detail_state(rid, router=None):
    """A ``RunDetailState`` wired into a real state tree whose router carries ``rid``.

    ``router`` lives on Reflex's *root* state and substates read it through their
    parent. A bare ``RunDetailState()`` has no parent, and how Reflex resolves
    ``router`` on a parent-less substate differs across versions (0.9.3 returns a
    default ``RouterData``; 0.9.12 dereferences ``parent_state`` and fails with
    ``'NoneType' object has no attribute 'rx_router_session'``). Building the
    same tree the app builds (root ``State`` -> substate) and assigning
    ``root.router`` is how Reflex itself populates the router, and behaves
    identically on both. ``router`` overrides the default ``RouterData`` (used to
    simulate a context with no usable router).
    """
    import reflex as rx
    from reflex.istate.data import RouterData

    root = rx.State(_reflex_internal_init=True)
    root.router = router if router is not None else RouterData.from_router_data(
        {"pathname": f"/runs/{rid}", "query": {"rid": rid}, "headers": {}, "ip": "127.0.0.1"})
    return root.substates[us.RunDetailState.get_name()]


# =============================================================================
# RunsState
# =============================================================================


class TestRunsStateLoad:
    def test_no_history_dir_reports_a_redacted_message(self, sandbox, monkeypatch):
        """Forbidden env override -> fallback default dir, which does not exist."""
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        s = us.RunsState()
        s.load_runs()
        assert s.runs == [] and s.loading is False
        assert s.error.startswith("No run history at") and "<redacted-path>" in s.error
        assert str(sandbox.home) not in s.error

    def test_rows_are_trimmed_to_the_table_shape(self, sandbox):
        _seed(sandbox, run_id="aaaaaaaa11112222", started_at="2026-09-02T00:00:00")
        _seed(sandbox, run_id="bbbb", status="failed", started_at="2026-09-01T00:00:00",
              duration_seconds=None, final_loss=None, model_name=None, dataset_info=None)
        s = us.RunsState()
        s.load_runs()
        assert s.error == "" and s.loading is False
        assert [r["run_id"] for r in s.runs] == ["aaaaaaaa11112222", "bbbb"]  # newest first
        first, second = s.runs
        assert first["run_id_short"] == "aaaaaaaa"
        assert (first["duration"], first["final_loss"]) == ("12s", "0.1235")
        assert second["run_id_short"] == "bbbb" and second["status"] == "failed"
        assert (second["duration"], second["final_loss"], second["model"], second["dataset"]) == (
            "-", "-", "-", "-")
        assert s.last_loaded_at.endswith("+00:00")

    def test_unparseable_numbers_degrade_to_dashes(self, sandbox):
        _seed(sandbox, duration_seconds="soon", final_loss=["x"])
        s = us.RunsState()
        s.load_runs()
        assert (s.runs[0]["duration"], s.runs[0]["final_loss"]) == ("-", "-")

    def test_missing_run_id_renders_a_dash(self, sandbox):
        _seed(sandbox, run_id=None)
        s = us.RunsState()
        s.load_runs()
        assert s.runs[0]["run_id_short"] == "-" and s.runs[0]["run_id"] == ""

    def test_status_filter_is_applied(self, sandbox):
        _seed(sandbox, run_id="r-done", status="completed")
        _seed(sandbox, run_id="r-bad", status="failed", started_at="2026-09-03T00:00:00")
        s = us.RunsState()
        s.status_filter = "failed"
        s.load_runs()
        assert [r["run_id"] for r in s.runs] == ["r-bad"]

    def test_bad_filter_value_surfaces_as_invalid_filter(self, sandbox):
        _seed(sandbox)
        s = us.RunsState()
        s.status_filter = "nonsense"  # set directly, bypassing the setter allowlist
        s.load_runs()
        assert s.runs == [] and s.error.startswith("Invalid filter:")

    def test_manager_failure_is_sanitised_and_split_into_message_and_hint(self, sandbox, monkeypatch):
        """Mocked: ``RunHistoryManager.list_runs`` raises a structured error."""
        from backpropagate.exceptions import BackpropagateError

        _seed(sandbox)

        def boom(self, status=None, limit=None):
            raise BackpropagateError(message="cannot parse /home/alice/run_history.json",
                                     code="STATE_HISTORY_CORRUPT", suggestion="restore from backup")

        monkeypatch.setattr(RunHistoryManager, "list_runs", boom)
        s = us.RunsState()
        s.load_runs()
        assert s.error.startswith("STATE_HISTORY_CORRUPT: ") and "alice" not in s.error
        assert s.error_suggestion == "restore from backup" and s.runs == []

    def test_foreign_manager_failure_is_opaque(self, sandbox, monkeypatch):
        _seed(sandbox)

        def boom(self, status=None, limit=None):
            raise OSError("[Errno 13] denied: '/home/alice/run_history.json'")

        monkeypatch.setattr(RunHistoryManager, "list_runs", boom)
        s = us.RunsState()
        s.load_runs()
        assert "alice" not in s.error and "Internal error during loading run history" in s.error
        assert s.error_suggestion == ""

    def test_missing_checkpoints_module_is_reported(self, sandbox, monkeypatch):
        """Mocked: ``backpropagate.checkpoints`` made un-importable."""
        import sys

        sandbox.out.mkdir(parents=True)
        monkeypatch.setitem(sys.modules, "backpropagate.checkpoints", None)
        s = us.RunsState()
        s.load_runs()
        assert s.error.startswith("checkpoints module unavailable") and s.runs == []
        assert s.loading is False

    def test_load_clears_a_previous_error(self, sandbox):
        _seed(sandbox)
        s = us.RunsState()
        s.error, s.error_suggestion = "stale", "stale hint"
        s.load_runs()
        assert (s.error, s.error_suggestion) == ("", "")


class TestRunsStateOverride:
    def test_benign_override_directory_is_read(self, sandbox, tmp_path):
        other = tmp_path / "elsewhere"
        mgr = RunHistoryManager(str(other))
        mgr.record_run({"run_id": "from-override", "status": "completed",
                        "started_at": "2026-01-01T00:00:00"})
        s = us.RunsState()
        s.output_dir_override = str(other)
        s.load_runs()
        assert [r["run_id"] for r in s.runs] == ["from-override"]

    @pytest.mark.parametrize("sub", [".ssh", ".aws", ".config"])
    def test_forbidden_override_set_directly_is_refused_without_reading(self, sandbox, sub, monkeypatch):
        import backpropagate.checkpoints as ck

        def tripwire(_d):
            raise AssertionError("RunHistoryManager must not be built for a forbidden base")

        monkeypatch.setattr(ck, "RunHistoryManager", tripwire)
        s = us.RunsState()
        s.output_dir_override = str(sandbox.home / sub)
        s.load_runs()
        assert s.runs == [] and "system or credential" in s.error
        assert str(sandbox.home) not in s.error  # redacted

    def test_guard_failure_fails_closed_to_the_sandbox_default(self, sandbox, tmp_path, monkeypatch):
        """Mocked: the forbidden-base check raises; the override must NOT be read."""
        import backpropagate.ui_security as sec

        _seed(sandbox, run_id="in-sandbox")
        other = tmp_path / "unvalidated"
        RunHistoryManager(str(other)).record_run(
            {"run_id": "in-override", "status": "completed", "started_at": "2026-01-01"})

        def boom(_p):
            raise RuntimeError("guard broke")

        monkeypatch.setattr(sec, "_is_forbidden_output_base", boom)
        s = us.RunsState()
        s.output_dir_override = str(other)
        s.load_runs()
        assert [r["run_id"] for r in s.runs] == ["in-sandbox"]

    def test_guard_and_default_resolution_both_failing_uses_documented_default(
        self, sandbox, monkeypatch
    ):
        import backpropagate.ui_security as sec

        _seed(sandbox, run_id="legacy")

        def boom(*_a, **_k):
            raise RuntimeError("down")

        monkeypatch.setattr(sec, "_is_forbidden_output_base", boom)
        monkeypatch.setattr(sec, "get_ui_output_dir", boom)
        s = us.RunsState()
        s.output_dir_override = str(sandbox.home / "anything")
        s.load_runs()
        assert [r["run_id"] for r in s.runs] == ["legacy"]

    def test_default_resolution_failure_without_override_uses_documented_default(
        self, sandbox, monkeypatch
    ):
        import backpropagate.ui_security as sec

        _seed(sandbox, run_id="legacy2")
        monkeypatch.setattr(sec, "get_ui_output_dir",
                            lambda: (_ for _ in ()).throw(RuntimeError("down")))
        s = us.RunsState()
        s.load_runs()
        assert [r["run_id"] for r in s.runs] == ["legacy2"]

    def test_set_output_dir_override_validates_through_the_sandbox(self, sandbox):
        s = us.RunsState()
        s.set_output_dir_override("/etc")
        assert s.output_dir_override == "" and s.error.startswith("Invalid path:")
        ok = str(sandbox.out / "custom")
        s.set_output_dir_override(ok)
        assert s.output_dir_override == ok

    def test_override_with_no_history_dir_reports_missing(self, sandbox, tmp_path):
        s = us.RunsState()
        s.output_dir_override = str(tmp_path / "does-not-exist")
        s.load_runs()
        assert s.error.startswith("No run history at")


class TestRunsStateFilterAndErrors:
    def test_setting_a_known_filter_reloads(self, sandbox):
        _seed(sandbox, run_id="ok1", status="completed")
        _seed(sandbox, run_id="bad1", status="failed", started_at="2026-09-05T00:00:00")
        s = us.RunsState()
        s.set_status_filter("completed")
        assert s.status_filter == "completed" and [r["run_id"] for r in s.runs] == ["ok1"]
        s.set_status_filter("")
        assert {r["run_id"] for r in s.runs} == {"ok1", "bad1"}

    def test_unknown_filter_is_dropped_logged_and_does_not_reload(self, sandbox, caplog):
        s = us.RunsState()
        s.status_filter = "running"
        s.runs = [{"run_id": "keep"}]
        with caplog.at_level(logging.WARNING, logger="backpropagate.ui_state"):
            s.set_status_filter("evil'; DROP")
        assert s.status_filter == "running" and s.runs == [{"run_id": "keep"}]
        assert any("unknown value" in r.getMessage() for r in caplog.records)

    def test_clear_error(self):
        s = us.RunsState()
        s.error, s.error_suggestion = "e", "h"
        s.clear_error()
        assert (s.error, s.error_suggestion) == ("", "")


# =============================================================================
# RunDetailState.load_run
# =============================================================================


class TestRunDetailLoad:
    def _make_checkpoints(self, root: Path):
        (root / "checkpoint-100").mkdir(parents=True)
        (root / "checkpoint-100" / "adapter.bin").write_bytes(b"x" * 2048)
        (root / "checkpoint-200").mkdir()
        (root / "checkpoint-200" / "nested").mkdir()
        (root / "checkpoint-200" / "nested" / "w.bin").write_bytes(b"y" * 1024)
        (root / "stray-file.txt").write_text("not a checkpoint dir", encoding="utf-8")
        (root / "training.log").write_text(
            "step 1 loaded /home/alice/.cache/huggingface/hub/model\nplain line\n", encoding="utf-8")

    def test_full_run_populates_every_section(self, sandbox, monkeypatch, tmp_path):
        cp = tmp_path / "ckpts"
        self._make_checkpoints(cp)
        _seed(sandbox, run_id="runfull1", checkpoint_path=str(cp),
              loss_history=[1.0, 0.5, "x", True, None], steps=100,
              hyperparameters={"lr": 0.0002}, dataset_info="/home/alice/data/train.jsonl",
              session_kind="single_run", completed_at="2026-09-01T10:05:00")
        s = _detail_state("runfull1")
        s.load_run()
        assert s.current_run_id == "runfull1" and s.error == "" and s.not_found is False
        assert s.loading is False and s.was_deleted is False
        assert (s.status, s.model, s.duration, s.final_loss) == (
            "completed", "Qwen/Qwen2.5-7B", "12s", "0.1235")
        assert s.completed_at == "2026-09-01T10:05:00"
        assert s.dataset == "train.jsonl"  # the file name, no home dir (ui-v2 P2)
        assert s.loss_history == [1.0, 0.5, 1.0]  # non-numeric dropped (bool is an int subclass)
        assert s.loss_chart_data[0] == {"step": 0, "loss": 1.0}
        hp = {row["key"]: row["value"] for row in s.hyperparameters}
        assert hp["steps"] == "100" and json.loads(hp["hyperparameters"]) == {"lr": 0.0002}
        assert "run_id" not in hp and "status" not in hp and "loss_history" not in hp
        assert [c["name"] for c in s.checkpoints] == ["checkpoint-100", "checkpoint-200"]
        assert s.checkpoints[0]["size_mb"] == "0.0" and s.checkpoints[1]["timestamp"]
        assert any("<redacted-path>" in line for line in s.log_lines)
        assert not any("alice" in line for line in s.log_lines)
        assert "plain line" in s.log_lines
        # the full checkpoint path stays server-side; the client-facing var is redacted
        assert s._checkpoint_path == str(cp)
        assert s.checkpoint_path_display == us._redact_action(str(cp))

    def test_long_values_are_truncated_in_the_hyperparameter_table(self, sandbox, monkeypatch):
        _seed(sandbox, run_id="longval", failure_reason="x" * 500,
              hyperparameters={"k": "v" * 500})
        s = _detail_state("longval")
        s.load_run()
        assert all(len(row["value"]) <= 200 for row in s.hyperparameters)

    def test_prefix_of_a_run_id_resolves(self, sandbox, monkeypatch):
        _seed(sandbox, run_id="abcdef123456")
        s = _detail_state("abcdef")
        s.load_run()
        assert s.status == "completed" and s.not_found is False

    def test_unknown_run_is_not_found_with_redacted_error(self, sandbox, monkeypatch):
        _seed(sandbox)
        s = _detail_state("missing1")
        s.load_run()
        assert s.not_found is True and "missing1" in s.error
        assert str(sandbox.home) not in s.error and s.loading is False

    def test_no_route_param_and_no_current_id_is_not_found(self, sandbox, monkeypatch):
        _seed(sandbox)
        s = _detail_state("")
        s.load_run()
        assert s.not_found is True

    def test_router_failure_falls_back_to_the_programmatic_id(self, sandbox, monkeypatch):
        _seed(sandbox, run_id="fallback1")

        from reflex.istate.data import RouterData

        class _BadRouter(RouterData):
            """A real ``RouterData`` (Reflex reads other attributes off it, e.g. the
            session) whose ``page`` lookup fails, as when no route is bound."""

            @property
            def page(self):
                raise RuntimeError("no router in this context")

        s = _detail_state("", router=_BadRouter())
        s.current_run_id = "fallback1"
        s.load_run()
        assert s.status == "completed" and s.not_found is False

    @pytest.mark.parametrize("rid", ["--to=/etc/passwd", "../../x", "a b", "-x", "a" * 70])
    def test_hostile_route_params_never_become_the_current_id(self, sandbox, monkeypatch, rid):
        s = _detail_state(rid)
        s.was_deleted = True
        s.load_run()
        assert s.current_run_id == "" and s.not_found is True
        assert s.error.startswith("Invalid run id") and s.was_deleted is False
        assert s.loading is False

    def test_missing_history_dir(self, sandbox, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        s = _detail_state("anything")
        s.load_run()
        assert s.error.startswith("No run history at") and "<redacted-path>" in s.error

    def test_checkpoints_module_failure(self, sandbox, monkeypatch):
        """Mocked: ``backpropagate.checkpoints`` made un-importable."""
        import sys

        sandbox.out.mkdir(parents=True)
        monkeypatch.setitem(sys.modules, "backpropagate.checkpoints", None)
        s = _detail_state("rid1")
        s.load_run()
        assert s.error.startswith("checkpoints module unavailable") and s.loading is False

    def test_entry_without_optional_fields_renders_dashes(self, sandbox, monkeypatch):
        _seed(sandbox, run_id="sparse", status=None, model_name=None, dataset_info=None,
              duration_seconds=None, final_loss=None, started_at=None, loss_history=None)
        s = _detail_state("sparse")
        s.load_run()
        assert (s.status, s.model, s.dataset, s.duration, s.final_loss) == ("-",) * 5
        assert s.loss_history == [] and s.checkpoints == [] and s.log_lines == []
        assert s.checkpoint_path_display == "-"
        assert s.started_at != ""  # falls back to the entry timestamp

    def test_garbage_numbers_and_non_list_loss_history(self, sandbox, monkeypatch):
        _seed(sandbox, run_id="garbage", duration_seconds="soon", final_loss="n/a",
              loss_history="not a list")
        s = _detail_state("garbage")
        s.load_run()
        assert (s.duration, s.final_loss) == ("-", "-") and s.loss_history == []

    def test_checkpoint_dir_walk_failure_is_tolerated(self, sandbox, monkeypatch, tmp_path):
        """Mocked: ``Path.iterdir`` raises ``OSError`` for the checkpoint dir."""
        cp = tmp_path / "ck"
        cp.mkdir()
        _seed(sandbox, run_id="walkfail", checkpoint_path=str(cp))
        real = Path.iterdir

        def boom(self):
            if self == cp:
                raise OSError("io error")
            return real(self)

        monkeypatch.setattr(Path, "iterdir", boom)
        s = _detail_state("walkfail")
        s.load_run()
        assert s.checkpoints == [] and s.status == "completed" and s.error == ""

    def test_unreadable_log_is_tolerated(self, sandbox, monkeypatch, tmp_path):
        """Mocked: ``open`` raises ``OSError`` for training.log."""
        cp = tmp_path / "ck2"
        cp.mkdir()
        (cp / "training.log").write_text("hello", encoding="utf-8")
        _seed(sandbox, run_id="logfail", checkpoint_path=str(cp))
        real_open = builtins.open

        def deny(path, *a, **k):
            if str(path).endswith("training.log"):
                raise OSError("denied")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        s = _detail_state("logfail")
        s.load_run()
        assert s.log_lines == [] and s.status == "completed"

    def test_log_tail_is_capped_at_200_lines(self, sandbox, monkeypatch, tmp_path):
        cp = tmp_path / "ck3"
        cp.mkdir()
        (cp / "training.log").write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")
        _seed(sandbox, run_id="biglog", checkpoint_path=str(cp))
        s = _detail_state("biglog")
        s.load_run()
        assert len(s.log_lines) == 200 and s.log_lines[-1] == "line 499"


# =============================================================================
# RunDetailState actions
# =============================================================================


class TestRunDetailActions:
    def _loaded(self, sandbox, monkeypatch, run_id="runact01", **entry):
        _seed(sandbox, run_id=run_id, **entry)
        s = _detail_state(run_id)
        s.load_run()
        return s

    # ---- diff_against -------------------------------------------------

    def test_diff_runs_shells_out_with_end_of_options_separator(self, monkeypatch):
        """Mocked: ``shutil.which`` + ``subprocess.run`` (the ``backprop`` CLI)."""
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))
            return SimpleNamespace(returncode=0, stdout="DIFF OUTPUT", stderr="")

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/backprop" if name == "backprop" else None)
        monkeypatch.setattr(subprocess, "run", fake_run)
        s = us.RunDetailState()
        s.current_run_id = "runA"
        s.action_error = "old error"
        s.diff_against("runB")
        assert calls[0][0] == ["/usr/bin/backprop", "diff-runs", "--", "runA", "runB"]
        assert calls[0][1]["timeout"] == 30 and calls[0][1]["check"] is False
        assert (s.action_result, s.action_error, s.action_in_flight) == ("DIFF OUTPUT", "", "")

    def test_diff_runs_nonzero_exit_surfaces_stderr(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/bin/backprop")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=2, stdout="", stderr="unknown run " + "z" * 2000))
        s = us.RunDetailState()
        s.current_run_id = "runA"
        s.action_result = "stale"
        s.diff_against("runB")
        assert s.action_error.startswith("unknown run") and len(s.action_error) == 1000
        assert s.action_result == "" and s.action_in_flight == ""

    def test_diff_runs_uses_the_alternate_binary_name(self, monkeypatch):
        seen = []
        monkeypatch.setattr("shutil.which", lambda name: "/bin/backpropagate" if name == "backpropagate" else None)
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: seen.append(cmd) or SimpleNamespace(
            returncode=0, stdout="", stderr=""))
        s = us.RunDetailState()
        s.current_run_id = "runA"
        s.diff_against("runB")
        assert seen[0][0] == "/bin/backpropagate"

    def test_diff_runs_without_a_cli_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        s = us.RunDetailState()
        s.current_run_id = "runA"
        s.diff_against("runB")
        assert "not found on PATH" in s.action_error and s.action_in_flight == ""

    @pytest.mark.parametrize("exc", [subprocess.TimeoutExpired(cmd="x", timeout=30),
                                     FileNotFoundError("/home/alice/bin/backprop")])
    def test_diff_runs_process_failures_are_redacted_and_release_the_flag(self, monkeypatch, exc):
        monkeypatch.setattr("shutil.which", lambda name: "/bin/backprop")

        def boom(*a, **k):
            raise exc

        monkeypatch.setattr(subprocess, "run", boom)
        s = us.RunDetailState()
        s.current_run_id = "runA"
        s.diff_against("runB")
        assert s.action_error.startswith("diff-runs failed:") and "alice" not in s.action_error
        assert s.action_in_flight == ""

    @pytest.mark.parametrize(("cur", "other"), [("", "x"), ("x", ""), ("", "")])
    def test_diff_requires_both_ids(self, cur, other):
        s = us.RunDetailState()
        s.current_run_id = cur
        s.diff_against(other)
        assert s.action_error == "Both run IDs are required for diff."

    @pytest.mark.parametrize(("cur", "other"), [("--evil", "ok"), ("ok", "--to=/etc/passwd"),
                                                ("ok", "../x")])
    def test_diff_revalidates_ids_at_the_sink_and_never_spawns(self, monkeypatch, cur, other):
        monkeypatch.setattr("shutil.which", lambda name: "/bin/backprop")

        def tripwire(*a, **k):
            raise AssertionError("subprocess must not be spawned with an unvalidated id")

        monkeypatch.setattr(subprocess, "run", tripwire)
        s = us.RunDetailState()
        s.current_run_id = cur
        s.diff_against(other)
        assert s.action_error.startswith("Invalid run id")

    def test_diff_with_input_branches(self, monkeypatch):
        s = us.RunDetailState()
        s.current_run_id = "runA"
        s.action_result = "old"
        s.diff_with_input()
        assert s.action_error == "Enter a comparison run id." and s.action_result == ""
        s.diff_other_run_id = "runA"
        s.action_result = "old"
        s.diff_with_input()
        assert "must differ" in s.action_error and s.action_result == ""
        calls = []
        monkeypatch.setattr("shutil.which", lambda name: "/bin/backprop")
        monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd) or SimpleNamespace(
            returncode=0, stdout="same", stderr=""))
        s.diff_other_run_id = "runB"
        s.diff_with_input()
        assert calls == [["/bin/backprop", "diff-runs", "--", "runA", "runB"]]
        assert s.action_result == "same"

    def test_set_diff_other_run_id_validates_and_clears_stale_error(self):
        s = us.RunDetailState()
        s.set_diff_other_run_id("--to=x")
        assert s.diff_other_run_id == "" and s.action_error.startswith("Invalid run id")
        s.set_diff_other_run_id("good_id")
        assert s.diff_other_run_id == "good_id" and s.action_error == ""
        s.action_error = "some other error"
        s.set_diff_other_run_id("good_id2")
        assert s.action_error == "some other error"  # only run-id errors are auto-cleared

    # ---- replay -------------------------------------------------------

    def test_replay_preflight_ok(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch, session_kind="multi_run")
        s.replay()
        assert s.action_error == "" and s.action_in_flight == ""
        assert "Dry-run OK" in s.action_result and "session=multi_run" in s.action_result
        assert "backprop replay runact01" in s.action_result

    def test_replay_requires_recorded_dataset(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch, dataset_info=None)
        s.replay()
        assert "no dataset_info" in s.action_error and s.action_result == ""

    def test_replay_unknown_run_and_validation(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)
        s.current_run_id = "ghost"
        s.replay()
        assert "not found" in s.action_error and str(sandbox.home) not in s.action_error
        s.current_run_id = ""
        s.replay()
        assert s.action_error == "No run loaded."
        s.current_run_id = "--evil"
        s.replay()
        assert s.action_error.startswith("Invalid run id")

    def test_replay_with_missing_history_dir(self, sandbox, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        s = us.RunDetailState()
        s.current_run_id = "runx"
        s.replay()
        assert s.action_error.startswith("No run history at") and s.action_in_flight == ""

    def test_replay_internal_failure_is_redacted(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)

        def boom(self, run_id):
            raise OSError("cannot read /home/alice/run_history.json")

        monkeypatch.setattr(RunHistoryManager, "get_run", boom)
        s.replay()
        assert s.action_error.startswith("replay check failed: OSError") and "alice" not in s.action_error
        assert s.action_in_flight == ""

    # ---- delete -------------------------------------------------------

    def test_delete_removes_the_entry_and_flips_was_deleted(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)
        s.delete_run()
        assert s.was_deleted is True and s.action_result == "Run runact01 deleted."
        assert s.action_error == "" and s.action_in_flight == ""
        assert RunHistoryManager(str(sandbox.out)).get_run("runact01") is None

    def test_delete_of_an_already_removed_run_reports_not_found(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)
        RunHistoryManager(str(sandbox.out)).delete_run("runact01")
        s.delete_run()
        assert s.was_deleted is False and "not found" in s.action_error

    def test_delete_guards(self, sandbox, monkeypatch):
        s = us.RunDetailState()
        s.delete_run()
        assert s.action_error == "No run loaded."
        s.current_run_id = "-x"
        s.delete_run()
        assert s.action_error.startswith("Invalid run id")
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        s.current_run_id = "runx"
        s.delete_run()
        assert s.action_error.startswith("No run history at")

    def test_delete_failure_is_redacted(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)

        def boom(self, run_id):
            raise PermissionError("denied /home/alice/run_history.json")

        monkeypatch.setattr(RunHistoryManager, "delete_run", boom)
        s.delete_run()
        assert s.action_error.startswith("delete failed: PermissionError")
        assert "alice" not in s.action_error and s.action_in_flight == ""

    # ---- export -------------------------------------------------------

    def test_export_writes_one_jsonl_record_inside_the_sandbox(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)
        s.export_run()
        assert s.action_error == "" and s.action_in_flight == ""
        files = list((sandbox.out / "exports").glob("run-runact01-*.jsonl"))
        assert len(files) == 1 and files[0].resolve().is_relative_to(sandbox.out)
        record = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert record["run_id"] == "runact01" and record["model_name"] == "Qwen/Qwen2.5-7B"
        assert s.action_result.startswith("Exported run runact01 to")
        assert str(sandbox.home) not in s.action_result  # path scrubbed before it reaches the client

    def test_export_guards_and_unknown_run(self, sandbox, monkeypatch):
        s = self._loaded(sandbox, monkeypatch)
        s.current_run_id = "ghost"
        s.export_run()
        assert "not found" in s.action_error
        s.current_run_id = ""
        s.export_run()
        assert s.action_error == "No run loaded."
        s.current_run_id = "../escape"
        s.export_run()
        assert s.action_error.startswith("Invalid run id")
        assert not list(sandbox.home.rglob("escape*"))

    def test_export_with_missing_history_dir(self, sandbox, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        s = us.RunDetailState()
        s.current_run_id = "runx"
        s.export_run()
        assert s.action_error.startswith("No run history at")

    def test_export_write_failure_is_redacted(self, sandbox, monkeypatch):
        """Mocked: ``open`` raises while writing the export file."""
        s = self._loaded(sandbox, monkeypatch)
        real_open = builtins.open

        def deny(path, *a, **k):
            if "exports" in str(path):
                raise OSError(f"disk full writing {path}")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        s.export_run()
        assert s.action_error.startswith("export failed: OSError")
        assert str(sandbox.home) not in s.action_error and s.action_in_flight == ""

    def test_clear_action_message(self):
        s = us.RunDetailState()
        s.action_result, s.action_error, s.action_error_suggestion = "r", "e", "h"
        s.clear_action_message()
        assert (s.action_result, s.action_error, s.action_error_suggestion) == ("", "", "")


# =============================================================================
# ModelsState
# =============================================================================


def _make_model(cache: Path, dir_name: str, nbytes: int):
    d = cache / dir_name / "snapshots" / "abc123"
    d.mkdir(parents=True)
    (d / "weights.bin").write_bytes(b"w" * nbytes)
    return cache / dir_name


class TestModelsStateLoad:
    def test_missing_cache_reports_a_redacted_message(self, sandbox):
        s = us.ModelsState()
        s.load_models()
        assert s.models == [] and s.total_size_label == "" and s.loading is False
        assert s.error.startswith("No Hugging Face cache at ~/") and str(sandbox.home) not in s.error
        assert s.cache_dir_display and str(sandbox.home) not in s.cache_dir_display
        assert s._cache_dir == str(sandbox.cache)

    def test_lists_models_largest_first_and_unmangles_names(self, sandbox):
        _make_model(sandbox.cache, "models--meta-llama--Llama-3.1-8B", 3 * 1024 * 1024)
        _make_model(sandbox.cache, "models--org--small", 1024)
        (sandbox.cache / "datasets--org--data").mkdir(parents=True)  # ignored: not a model
        (sandbox.cache / "stray.txt").write_text("x", encoding="utf-8")  # ignored: not a dir
        s = us.ModelsState()
        s.load_models()
        assert s.error == "" and s.loading is False
        assert [m["name"] for m in s.models] == ["meta-llama/Llama-3.1-8B", "org/small"]
        assert s.models[0]["dir_name"] == "models--meta-llama--Llama-3.1-8B"
        assert s.models[0]["size_mb"] == "3.0" and s.models[1]["size_mb"] == "0.0"
        assert s.total_size_label == "3.0 MB" and s.last_loaded_at
        assert s.models[0]["last_modified"] != "-"

    def test_name_with_single_separator_unmangles_once(self, sandbox):
        _make_model(sandbox.cache, "models--solo", 10)
        s = us.ModelsState()
        s.load_models()
        assert s.models[0]["name"] == "solo"

    def test_walk_failure_is_reported_redacted(self, sandbox, monkeypatch):
        """Mocked: ``Path.iterdir`` raises ``OSError`` for the cache dir."""
        sandbox.cache.mkdir(parents=True)
        real = Path.iterdir

        def boom(self):
            if self == sandbox.cache:
                raise OSError(f"cannot list {self}")
            return real(self)

        monkeypatch.setattr(Path, "iterdir", boom)
        s = us.ModelsState()
        s.load_models()
        assert s.error.startswith("Cannot walk HF cache") and str(sandbox.home) not in s.error
        assert s.models == [] and s.total_size_label == "" and s.loading is False

    def test_unstatable_entries_count_as_zero_bytes_and_unknown_mtime(self, sandbox, monkeypatch):
        """Mocked: after the model dir is recognised, ``Path.rglob`` raises and ``Path.stat``
        then fails for that dir (a model being deleted mid-listing)."""
        model = _make_model(sandbox.cache, "models--org--flaky", 5000)
        real_rglob, real_stat = Path.rglob, Path.stat
        armed = {"on": False}

        def rglob_boom(self, pattern):
            if self == model:
                armed["on"] = True
                raise OSError("walk failed")
            return real_rglob(self, pattern)

        def stat_boom(self, *a, **k):
            if armed["on"] and self == model:
                raise OSError("stat failed")
            return real_stat(self, *a, **k)

        monkeypatch.setattr(Path, "rglob", rglob_boom)
        monkeypatch.setattr(Path, "stat", stat_boom)
        s = us.ModelsState()
        s.load_models()
        row = s.models[0]
        assert row["size_bytes"] == 0 and row["last_modified"] == "-"

    def test_cache_dir_display_empty_before_load(self):
        assert us.ModelsState().cache_dir_display == ""


class TestModelsStateDelete:
    @pytest.mark.parametrize(
        "name",
        ["", "random", "models--a/b", "models--a\\b", "models--..", "../models--x",
         "models--a/../../b", "datasets--x"],
    )
    def test_invalid_names_are_refused(self, sandbox, name):
        _make_model(sandbox.cache, "models--org--keep", 10)
        s = us.ModelsState()
        s.delete_model(name)
        assert s.error.startswith("Invalid model directory name")
        assert (sandbox.cache / "models--org--keep").exists()
        assert s.deleting_dir == ""

    def test_delete_removes_the_dir_reloads_and_releases_the_flag(self, sandbox):
        _make_model(sandbox.cache, "models--org--gone", 100)
        _make_model(sandbox.cache, "models--org--stay", 100)
        s = us.ModelsState()
        s.load_models()
        s.delete_model("models--org--gone")
        assert not (sandbox.cache / "models--org--gone").exists()
        assert (sandbox.cache / "models--org--stay").exists()
        assert [m["dir_name"] for m in s.models] == ["models--org--stay"]
        assert s.deleting_dir == "" and s.error == ""

    def test_missing_target_reports_not_found(self, sandbox):
        sandbox.cache.mkdir(parents=True)
        s = us.ModelsState()
        s.delete_model("models--org--never")
        assert s.error.startswith("Model directory not found") and s.deleting_dir == ""
        assert str(sandbox.home) not in s.error

    def test_reentrant_delete_of_the_same_target_is_a_silent_noop(self, sandbox):
        _make_model(sandbox.cache, "models--org--busy", 10)
        s = us.ModelsState()
        s.deleting_dir = "models--org--busy"
        s.delete_model("models--org--busy")
        assert (sandbox.cache / "models--org--busy").exists() and s.error == ""
        assert s.deleting_dir == "models--org--busy"  # the in-flight owner clears it

    def test_symlinked_cache_entry_is_refused_not_followed(self, sandbox, tmp_path):
        victim = tmp_path / "important"
        victim.mkdir()
        (victim / "keep.txt").write_text("precious", encoding="utf-8")
        sandbox.cache.mkdir(parents=True)
        link = sandbox.cache / "models--evil--link"
        try:
            link.symlink_to(victim, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")
        s = us.ModelsState()
        s.delete_model("models--evil--link")
        assert "symlinked cache entry" in s.error and str(sandbox.home) not in s.error
        assert (victim / "keep.txt").read_text(encoding="utf-8") == "precious"
        assert s.deleting_dir == ""

    def test_target_resolving_outside_the_cache_is_refused(self, sandbox, tmp_path, monkeypatch):
        """Mocked: ``Path.resolve`` maps the target outside the cache (a junction-style escape)."""
        outside = tmp_path / "outside-the-cache"
        outside.mkdir()
        (outside / "keep.txt").write_text("x", encoding="utf-8")
        _make_model(sandbox.cache, "models--org--escape", 10)
        real = Path.resolve

        def sneaky(self, *a, **k):
            if self.name == "models--org--escape":
                return outside
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", sneaky)
        s = us.ModelsState()
        s.delete_model("models--org--escape")
        assert s.error.startswith("Refusing to delete outside HF cache")
        assert (outside / "keep.txt").exists() and s.deleting_dir == ""

    def test_rmtree_failure_is_reported_redacted_and_releases_the_flag(self, sandbox, monkeypatch):
        """Mocked: ``shutil.rmtree`` raises ``OSError``."""
        _make_model(sandbox.cache, "models--org--locked", 10)

        def boom(path, *a, **k):
            raise PermissionError(f"in use: {path}")

        monkeypatch.setattr("shutil.rmtree", boom)
        s = us.ModelsState()
        s.delete_model("models--org--locked")
        assert s.error.startswith("Failed to delete models--org--locked")
        assert str(sandbox.home) not in s.error and s.deleting_dir == ""
        assert (sandbox.cache / "models--org--locked").exists()


# =============================================================================
# DatasetState.handle_upload
# =============================================================================


class _Reader:
    """Chunked async reader shaped like ``rx.upload``'s file objects."""

    def __init__(self, data: bytes, filename="data.jsonl"):
        self.filename = filename
        self._data = data
        self._pos = 0

    async def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk


class _OneShotReader:
    """Reader whose ``read`` takes no size argument (forces the TypeError fallback)."""

    def __init__(self, data: bytes, filename="data.jsonl"):
        self.filename = filename
        self._data = data

    async def read(self):
        return self._data


class _FailingReader:
    def __init__(self, exc, filename="data.jsonl", fail_on_chunked=True):
        self.filename = filename
        self.exc = exc
        self.fail_on_chunked = fail_on_chunked

    async def read(self, n=None):
        raise self.exc


_SHAREGPT = (
    '{"conversations": [{"from": "human", "value": "hi there"}, {"from": "gpt", "value": "hello!"}]}\n'
    '{"conversations": [{"from": "human", "value": "bye"}, {"from": "gpt", "value": "see you"}]}\n'
)


def _upload(state, files):
    return asyncio.run(state.handle_upload(files))


class TestHandleUploadGuards:
    @pytest.mark.parametrize("payload", [[], None, "x", {"a": 1}, ()])
    def test_non_list_or_empty_payloads_are_refused(self, sandbox, payload):
        s = us.DatasetState()
        _upload(s, payload)
        assert s.upload_error == "No files received" and s.upload_count == 0

    def test_multi_file_drops_are_rejected_whole(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(b"{}"), _Reader(b"{}")])
        assert "Only one file per upload" in s.upload_error
        assert s.upload_count == 0 and not (sandbox.out / "uploads").exists()

    def test_per_session_cap(self, sandbox):
        s = us.DatasetState()
        s.upload_count = 5
        _upload(s, [_Reader(b"{}")])
        assert "Per-session upload cap reached (5 files)" in s.upload_error
        assert s.upload_count == 5

    def test_unresolvable_output_dir_is_sanitised(self, sandbox, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        s = us.DatasetState()
        _upload(s, [_Reader(b"{}")])
        assert s.upload_error.startswith("UI_OUTPUT_DIR_FORBIDDEN: ")
        assert str(sandbox.home) not in s.upload_error and s.upload_count == 0


class TestHandleUploadHappyPaths:
    def test_jsonl_is_persisted_and_stats_are_computed(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(_SHAREGPT.encode(), "train.jsonl")])
        target = sandbox.out / "uploads" / "train.jsonl"
        assert target.read_text(encoding="utf-8") == _SHAREGPT
        assert s.upload_error == "" and s.upload_count == 1
        assert s._uploaded_path == str(target) and s.has_upload is True
        assert s.uploaded_basename == "train.jsonl"
        assert s.record_count == 2
        assert s.avg_tokens > 0 and s.detected_format != ""
        public = {k: str(getattr(s, k)) for k in s.get_fields() if not k.startswith("_")}
        assert all(str(sandbox.home) not in v for v in public.values())

    def test_malformed_jsonl_lines_are_skipped_in_the_stats(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(b'{"text": "a"}\nthis line is not json\n\n{"text": "b"}\n', "m.jsonl")])
        assert s.upload_error == "" and s.record_count == 2

    def test_json_list_and_json_object(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(json.dumps([{"text": "a"}, {"text": "b"}, {"text": "c"}]).encode(), "a.json")])
        assert s.record_count == 3 and s.upload_error == ""
        s2 = us.DatasetState()
        _upload(s2, [_Reader(json.dumps({"text": "only"}).encode(), "b.json")])
        assert s2.record_count == 1

    def test_csv_is_accepted_without_stats(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(b"a,b\n1,2\n", "t.csv")])
        assert s.upload_error == "" and s._uploaded_path.endswith("t.csv")
        assert s.upload_count == 1
        # The page says why there is no preview, and that the file is still usable.
        assert "Only .jsonl and .json" in s.inspect_note
        assert "can still be used for training" in s.inspect_note
        assert (s.record_count, s.preview_records, s.cleanup_summary) == (0, [], "")

    def test_missing_filename_falls_back_to_unnamed(self, sandbox):
        s = us.DatasetState()
        r = _Reader(b"{}")
        r.filename = None
        _upload(s, [r])
        # "unnamed" has no extension -> rejected by the extension allowlist
        assert s.upload_error.startswith("Rejected unnamed:") and s.upload_count == 0

    def test_chunked_reader_streams_across_multiple_reads(self, sandbox):
        # one ~1.5 MB record -> two 1 MB reads, but a single stats row (a many-record
        # file of an undetectable format would log one warning per record)
        big = ('{"text": "' + "x" * 1_500_000 + '"}\n').encode()
        s = us.DatasetState()
        _upload(s, [_Reader(big, "big.jsonl")])
        assert s.upload_error == "" and s.record_count == 1
        assert (sandbox.out / "uploads" / "big.jsonl").stat().st_size == len(big)

    def test_one_shot_reader_uses_the_fallback_read(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_OneShotReader(b'{"text": "a"}\n', "one.jsonl")])
        assert s.upload_error == "" and (sandbox.out / "uploads" / "one.jsonl").exists()

    def test_stats_failure_never_blocks_the_upload(self, sandbox, monkeypatch):
        """Mocked: reading the file for the preview raises."""
        import backpropagate.dataset_prep as prep

        def boom(*a, **k):
            raise RuntimeError(f"stats exploded in {sandbox.home}")

        monkeypatch.setattr(prep, "summarise_dataset", boom)
        s = us.DatasetState()
        _upload(s, [_Reader(b'{"text": "a"}\n', "ok.jsonl")])
        assert s.upload_error == "" and s.upload_count == 1 and s.record_count == 0
        assert "could not be read for a preview" in s.inspect_note
        assert str(sandbox.home) not in s.inspect_note and "exploded" not in s.inspect_note

    def test_temp_file_cleanup_failure_is_tolerated(self, sandbox, monkeypatch):
        """Mocked: ``Path.unlink`` raises ``OSError`` while cleaning the staging file."""
        real = Path.unlink

        def boom(self, *a, **k):
            if self.suffix == ".jsonl" and "uploads" not in str(self):
                raise OSError("busy")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", boom)
        s = us.DatasetState()
        _upload(s, [_Reader(b'{"text": "a"}\n', "ok.jsonl")])
        assert s.upload_error == "" and s.upload_count == 1


class TestHandleUploadAdversarial:
    @pytest.mark.parametrize("name", ["evil.exe", "page.html", "x.svg", "run.py", "a.sh", "noext"])
    def test_dangerous_or_unknown_extensions_are_rejected_and_not_persisted(self, sandbox, name):
        s = us.DatasetState()
        _upload(s, [_Reader(b"MZ\x90\x00 payload", name)])
        assert s.upload_error.startswith(f"Rejected {name}:")
        assert s.upload_count == 0 and s._uploaded_path == ""
        uploads = sandbox.out / "uploads"
        assert not list(uploads.iterdir())

    @pytest.mark.parametrize(
        "name",
        ["../../evil.jsonl", "..\\..\\evil.jsonl", "/etc/cron.d/x.jsonl", "a/b/c.jsonl", "..", "...jsonl"],
    )
    def test_traversal_filenames_land_inside_uploads_only(self, sandbox, name):
        s = us.DatasetState()
        _upload(s, [_Reader(b'{"text": "a"}\n', name)])
        uploads = (sandbox.out / "uploads").resolve()
        written = [p for p in sandbox.home.rglob("*") if p.is_file()]
        assert all(p.resolve().is_relative_to(uploads) for p in written), written
        if s.upload_error == "":
            assert Path(s._uploaded_path).resolve().parent == uploads

    def test_oversize_stream_is_aborted_mid_flight_and_nothing_is_kept(self, sandbox, monkeypatch):
        import backpropagate.ui_security as sec

        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "max_upload_size_mb", 1)
        s = us.DatasetState()
        _upload(s, [_Reader(b"{" + b"x" * (3 * 1024 * 1024), "huge.jsonl")])
        assert "exceeds 1 MB cap (aborted mid-stream)" in s.upload_error
        assert s.upload_count == 0 and not list((sandbox.out / "uploads").iterdir())

    def test_oversize_one_shot_read_is_rejected(self, sandbox, monkeypatch):
        import backpropagate.ui_security as sec

        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "max_upload_size_mb", 1)
        s = us.DatasetState()
        _upload(s, [_OneShotReader(b"{" + b"x" * (2 * 1024 * 1024), "huge.jsonl")])
        assert "exceeds 1 MB cap" in s.upload_error and "mid-stream" not in s.upload_error
        assert s.upload_count == 0

    def test_exactly_at_the_cap_is_accepted(self, sandbox, monkeypatch):
        import backpropagate.ui_security as sec

        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "max_upload_size_mb", 1)
        s = us.DatasetState()
        payload = b'{"t": "' + b"x" * (1024 * 1024 - 9) + b'"}'
        assert len(payload) == 1024 * 1024
        _upload(s, [_Reader(payload, "edge.jsonl")])
        assert s.upload_error == "" and s.upload_count == 1

    def test_one_shot_read_failure_is_sanitised(self, sandbox):
        class _BadOneShot(_OneShotReader):
            async def read(self):
                raise OSError("socket closed reading /home/alice/tmp/up")

        s = us.DatasetState()
        _upload(s, [_BadOneShot(b"", "x.jsonl")])
        assert "alice" not in s.upload_error and "Internal error during reading upload" in s.upload_error
        assert s.upload_count == 0

    def test_streaming_read_failure_is_sanitised(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_FailingReader(OSError("reset by peer /home/alice/spool"), "x.jsonl")])
        assert "alice" not in s.upload_error and "Internal error during reading upload" in s.upload_error

    def test_magic_byte_spoof_is_rejected_when_enabled(self, sandbox, monkeypatch):
        import backpropagate.ui_security as sec

        monkeypatch.setattr(sec.DEFAULT_SECURITY_CONFIG, "validate_file_magic", True)
        s = us.DatasetState()
        _upload(s, [_Reader(b"<!DOCTYPE html><html><script>alert(1)</script>", "spoof.jsonl")])
        assert s.upload_error.startswith("Rejected spoof.jsonl:") and "magic-bytes" in s.upload_error
        assert s.upload_count == 0 and not list((sandbox.out / "uploads").iterdir())

    def test_failed_upload_does_not_erase_a_previous_good_one(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(b'{"text": "a"}\n', "good.jsonl")])
        good = s._uploaded_path
        _upload(s, [_Reader(b"MZ", "bad.exe")])
        assert s._uploaded_path == good and s.upload_count == 1 and s.upload_error

    def test_success_clears_a_previous_error(self, sandbox):
        s = us.DatasetState()
        _upload(s, [_Reader(b"MZ", "bad.exe")])
        assert s.upload_error
        _upload(s, [_Reader(b'{"text": "a"}\n', "good.jsonl")])
        assert s.upload_error == ""

    def test_five_successful_uploads_then_the_cap_applies(self, sandbox):
        s = us.DatasetState()
        for i in range(5):
            _upload(s, [_Reader(b'{"text": "a"}\n', f"f{i}.jsonl")])
        assert s.upload_count == 5 and s.upload_error == ""
        _upload(s, [_Reader(b'{"text": "a"}\n', "f6.jsonl")])
        assert "Per-session upload cap" in s.upload_error and s.upload_count == 5
        assert not (sandbox.out / "uploads" / "f6.jsonl").exists()
