"""Regression tests for three ``ui_state`` defects found while writing the UI coverage suite.

Each test failed on ``main`` before the matching fix:

1. Numeric setters: ``_coerce_int`` raised ``OverflowError`` / ``ValueError`` for
   ``"inf"`` / ``"1e999"`` / a float ``nan`` (an unhandled exception inside a
   Reflex event handler, triggered by typing into a number box), and
   ``_coerce_float("nan")`` returned ``nan``, which sailed through the clamp
   (``nan < lo`` and ``nan > hi`` are both False) and was stored as the learning
   rate / dropout / replay fraction.
2. ``RunsState.load_runs`` read ``raw["model"]`` / ``raw["dataset"]`` but
   ``RunHistoryManager`` stores ``model_name`` / ``dataset_info``, so the /runs
   table showed ``-`` in the Model and Dataset columns for every real run. (The
   CLI's ``list-runs`` does its own ``model_name`` -> ``model`` mapping.) The
   dataset value is usually an absolute path, so it is now redacted like the
   run-detail page's.
3. The /runs status filter offered ``Interrupted``; ``RunHistoryManager`` only
   knows ``running`` / ``completed`` / ``failed`` and raised ``ValueError`` for
   it, so choosing that option always produced an "Invalid filter" error.

Nothing is mocked; the run history is the real on-disk store under ``tmp_path``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from backpropagate import ui_state as us  # noqa: E402
from backpropagate.checkpoints import RunHistoryManager  # noqa: E402


class TestNonFiniteNumbersAreRejectedNotCrashed:
    @pytest.mark.parametrize(
        "raw", ["inf", "-inf", "Infinity", "1e999", "-1e999", "nan", "NaN",
                float("inf"), float("-inf"), float("nan")],
    )
    def test_coerce_int_returns_none_for_non_finite_values(self, raw):
        assert us._coerce_int(raw) is None

    @pytest.mark.parametrize("raw", ["nan", "NaN", float("nan")])
    def test_coerce_float_rejects_nan(self, raw):
        assert us._coerce_float(raw) is None

    @pytest.mark.parametrize("raw", ["inf", "-inf", float("inf"), float("-inf")])
    def test_coerce_float_still_lets_infinities_reach_the_clamp(self, raw):
        value = us._coerce_float(raw)
        assert value is not None and math.isinf(value)

    @pytest.mark.parametrize("raw", ["inf", "1e999", "-1e999", float("nan"), "nan"])
    def test_int_setters_report_an_error_and_keep_the_previous_value(self, raw):
        s = us.TrainState()
        s.set_steps(500)
        s.set_steps(raw)  # must not raise
        assert s.steps == 500 and "must be an integer" in s.steps_error
        m = us.MultiRunState()
        m.set_num_runs(7)
        m.set_num_runs(raw)
        assert m.num_runs == 7 and m.num_runs_error

    @pytest.mark.parametrize("handler", ["set_learning_rate", "set_lora_dropout"])
    def test_nan_is_not_stored_by_float_setters(self, handler):
        s = us.TrainState()
        before = (s.learning_rate, s.lora_dropout)
        getattr(s, handler)("nan")
        assert (s.learning_rate, s.lora_dropout) == before
        assert not math.isnan(s.learning_rate) and not math.isnan(s.lora_dropout)
        assert "must be a number" in getattr(s, f"{'learning_rate' if 'rate' in handler else 'lora_dropout'}_error")

    def test_replay_fraction_rejects_nan_and_clamps_infinity(self):
        m = us.MultiRunState()
        m.set_replay_fraction(0.5)
        m.set_replay_fraction("nan")
        assert m.replay_fraction == 0.5 and m.replay_fraction_error
        m.set_replay_fraction("inf")
        assert m.replay_fraction == 1.0 and "maximum" in m.replay_fraction_error

    def test_dataset_token_bounds_reject_non_finite(self):
        d = us.DatasetState()
        d.set_max_tokens("1e999")
        assert d.max_tokens == 2048 and d.max_tokens_error


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    out = (home / ".backpropagate" / "ui-outputs").resolve()
    out.mkdir(parents=True)
    return out


class TestRunsTableShowsModelAndDataset:
    def test_model_and_dataset_columns_come_from_the_stored_entry(self, sandbox):
        RunHistoryManager(str(sandbox)).record_run({
            "run_id": "run00001", "status": "completed", "started_at": "2026-09-01T10:00:00",
            "model_name": "Qwen/Qwen2.5-7B-Instruct", "dataset_info": "train.jsonl"})
        s = us.RunsState()
        s.load_runs()
        assert s.runs[0]["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert s.runs[0]["dataset"] == "train.jsonl"

    def test_absolute_dataset_paths_are_redacted_before_reaching_the_client(self, sandbox):
        RunHistoryManager(str(sandbox)).record_run({
            "run_id": "run00002", "status": "completed", "started_at": "2026-09-01T10:00:00",
            "model_name": "m", "dataset_info": "/home/alice/data/train.jsonl"})
        s = us.RunsState()
        s.load_runs()
        # ui-v2 P2: the dataset's file name, never the home dir / username.
        assert s.runs[0]["dataset"] == "train.jsonl"

    def test_entries_without_the_fields_still_render_dashes(self, sandbox):
        RunHistoryManager(str(sandbox)).record_run({
            "run_id": "run00003", "status": "failed", "started_at": "2026-09-01T10:00:00"})
        s = us.RunsState()
        s.load_runs()
        assert (s.runs[0]["model"], s.runs[0]["dataset"]) == ("-", "-")

    def test_legacy_short_keys_are_still_honoured(self, sandbox, monkeypatch):
        """A manager/fixture that already supplies ``model`` / ``dataset`` keeps working."""
        monkeypatch.setattr(RunHistoryManager, "list_runs", lambda self, status=None, limit=None: [
            {"run_id": "legacy01", "status": "completed", "model": "old-model", "dataset": "old.jsonl"}])
        s = us.RunsState()
        s.load_runs()
        assert (s.runs[0]["model"], s.runs[0]["dataset"]) == ("old-model", "old.jsonl")


class TestStatusFilterOffersOnlyWhatTheStoreAccepts:
    def test_every_filter_value_is_accepted_by_the_history_manager(self, sandbox):
        accepted = set(RunHistoryManager.VALID_STATUSES) | {""}
        assert set(us.RunsState._STATUS_FILTER_VALUES) <= accepted

    def test_every_filter_option_loads_without_an_error(self, sandbox):
        RunHistoryManager(str(sandbox)).record_run({
            "run_id": "r1", "status": "completed", "started_at": "2026-09-01T10:00:00"})
        s = us.RunsState()
        for value in us.RunsState._STATUS_FILTER_VALUES:
            s.set_status_filter(value)
            assert s.error == "", f"filter {value!r} produced: {s.error!r}"

    def test_the_runs_page_select_offers_only_accepted_values(self):
        from backpropagate.ui_app.pages.runs import runs_page

        rendered = str(runs_page().render())
        for status in ("running", "completed", "failed"):
            assert f'value:\\"{status}\\"' in rendered or status in rendered
        assert 'value:\\"interrupted\\"' not in rendered and "Interrupted" not in rendered
