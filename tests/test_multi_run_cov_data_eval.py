"""Dataset loading and the eval-gate plumbing of ``MultiRunTrainer``.

Real: dataset files on ``tmp_path`` through the real ``DatasetLoader``, real
``datasets.Dataset`` objects, real tiny PEFT adapter (config written by PEFT
itself), real ``torch.save``, real run-history bookkeeping. Mocked boundaries
(named per test): the HF Hub ``datasets.load_dataset`` and
``evaluate_run`` (which loads the base model from the Hub and generates text).
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from datasets import Dataset

from backpropagate import multi_run
from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy, RunHistoryManager
from backpropagate.config import settings
from backpropagate.datasets import DatasetLoader
from backpropagate.eval import EvalResult
from backpropagate.exceptions import BackpropagateError, DatasetError, DatasetNotFoundError
from backpropagate.multi_run import MultiRunConfig, MultiRunTrainer
from tests.test_multi_run_cov_support import FakeInnerTrainer, build_peft_llama, text_dataset

MR_LOGGER = "backpropagate.multi_run"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _mrt(**cfg):
    return MultiRunTrainer(model="m", config=MultiRunConfig(**cfg))


def _alpaca_row(i, output=None):
    output = f"a{i}" if output is None else output
    return {"instruction": f"q{i}", "input": "", "output": output}


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


# =============================================================================
# _load_full_dataset
# =============================================================================


class TestLoadFullDataset:
    def test_dataset_object_is_returned_as_is(self):
        ds = text_dataset(5)
        assert _mrt()._load_full_dataset(ds) is ds

    def test_unsupported_type_is_a_dataset_error(self):
        with pytest.raises(DatasetError) as exc_info:
            _mrt()._load_full_dataset(3.14)
        assert "Unsupported dataset type: float" in str(exc_info.value)

    def test_missing_local_file_is_dataset_not_found(self, tmp_path):
        with pytest.raises(DatasetNotFoundError):
            _mrt()._load_full_dataset(str(tmp_path / "absent.jsonl"))

    def test_local_file_with_warnings_and_errors_still_loads_with_logs(self, tmp_path, caplog):
        """One empty reply (a validation *warning*) and one unrecognisable row (an
        *error*): the load proceeds and reports both."""
        rows = [_alpaca_row(i) for i in range(6)]
        rows.append(_alpaca_row(6, output=""))  # -> empty_content warning
        rows.append({"unrelated": "row"})       # -> unknown format error
        path = _write_jsonl(tmp_path / "data.jsonl", rows)

        with caplog.at_level(logging.INFO, logger=MR_LOGGER):
            ds = _mrt()._load_full_dataset(path)

        msgs = [r.getMessage() for r in caplog.records]
        assert any("DatasetLoader detected format: alpaca" in m for m in msgs)
        assert any(m.startswith("Dataset validation warning:") for m in msgs)
        assert any("validation errors" in m and "proceeding anyway" in m for m in msgs)
        assert any(m.startswith("  Row ") for m in msgs)  # per-error detail lines
        assert len(ds) >= 6 and "text" in ds.column_names

    def test_dataset_loader_instance_is_converted_and_its_warnings_logged(
        self, tmp_path, caplog
    ):
        rows = [_alpaca_row(i) for i in range(6)] + [_alpaca_row(6, output="")]
        loader = DatasetLoader(_write_jsonl(tmp_path / "d.jsonl", rows), validate=True)
        assert loader.validation_result.warnings  # precondition for the log branch

        with caplog.at_level(logging.WARNING, logger=MR_LOGGER):
            ds = _mrt()._load_full_dataset(loader)

        assert len(ds) == len(loader.to_hf_dataset())
        assert any("Dataset validation warning:" in r.getMessage() for r in caplog.records)

    def test_hub_name_goes_through_load_dataset_with_the_configured_split(self, monkeypatch):
        """Mocked: ``datasets.load_dataset`` (HF Hub network call)."""
        calls = []
        expected = text_dataset(7)

        def fake_load(name, split=None, **kw):
            calls.append((name, split))
            return expected

        monkeypatch.setattr("datasets.load_dataset", fake_load)

        assert _mrt()._load_full_dataset("org/some-dataset") is expected
        assert calls == [("org/some-dataset", settings.data.dataset_split)]

    def test_none_falls_back_to_the_configured_default_dataset(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "datasets.load_dataset", lambda name, split=None, **kw: calls.append(name) or text_dataset(2)
        )
        _mrt()._load_full_dataset(None)
        assert calls == [settings.data.dataset_name]

    def test_unexpected_loader_failure_is_wrapped(self, monkeypatch):
        def boom(*_a, **_k):
            raise ValueError("schema mismatch")

        monkeypatch.setattr("datasets.load_dataset", boom)

        with pytest.raises(DatasetError) as exc_info:
            _mrt()._load_full_dataset("org/broken")

        assert "Failed to load dataset: schema mismatch" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_file_not_found_from_the_hub_loader_is_dataset_not_found(self, monkeypatch):
        def missing(*_a, **_k):
            raise FileNotFoundError("no such dataset script")

        monkeypatch.setattr("datasets.load_dataset", missing)

        with pytest.raises(DatasetNotFoundError):
            _mrt()._load_full_dataset("org/gone")


# =============================================================================
# held-out derivation / adapter config / references
# =============================================================================


class TestDeriveHeldout:
    def test_none_dataset(self):
        with pytest.raises(BackpropagateError) as exc_info:
            _mrt()._derive_last_decile_heldout(None)
        assert exc_info.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"

    def test_empty_dataset(self):
        with pytest.raises(BackpropagateError) as exc_info:
            _mrt()._derive_last_decile_heldout(Dataset.from_dict({"text": []}))
        assert "dataset is empty" in str(exc_info.value)

    def test_last_ten_percent_of_rows_in_order(self):
        ds = Dataset.from_dict({"text": [f"row{i}" for i in range(25)]})
        assert _mrt()._derive_last_decile_heldout(ds) == ["row23", "row24"]

    def test_tiny_dataset_still_yields_one_row(self):
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        assert _mrt()._derive_last_decile_heldout(ds) == ["c"]

    def test_plain_sequence_without_select_is_indexed_and_stripped(self):
        rows = [{"text": "x"}] * 8 + [{"text": "  last  "}, "raw string row"]
        assert _mrt()._derive_last_decile_heldout(rows) == ["raw string row"]

    def test_rows_that_flatten_to_nothing_are_rejected(self):
        ds = Dataset.from_dict({"text": ["ok"] * 9 + ["   "]})
        with pytest.raises(BackpropagateError) as exc_info:
            _mrt()._derive_last_decile_heldout(ds)
        assert "zero usable text rows" in str(exc_info.value)


class TestWriteEvalAdapterConfig:
    def test_live_peft_config_is_persisted(self, tmp_path):
        mrt = _mrt()
        mrt._trainer = FakeInnerTrainer(build_peft_llama(r=4))

        mrt._write_eval_adapter_config(tmp_path)

        cfg = json.loads((tmp_path / "adapter_config.json").read_text())
        assert cfg["r"] == 4 and cfg["lora_alpha"] == 8
        assert set(cfg["target_modules"]) == {"q_proj", "v_proj"}

    def test_first_config_is_used_when_the_active_adapter_name_is_unknown(self, tmp_path):
        """Stand-in model: ``peft_config`` keyed by a name that is not the active one."""
        from peft import LoraConfig

        model = types.SimpleNamespace(
            peft_config={"other": LoraConfig(r=2, lora_alpha=3)}, active_adapter="missing"
        )
        mrt = _mrt()
        mrt._trainer = types.SimpleNamespace(_model=model)

        mrt._write_eval_adapter_config(tmp_path)

        cfg = json.loads((tmp_path / "adapter_config.json").read_text())
        assert cfg["r"] == 2 and cfg["lora_alpha"] == 3

    def test_minimal_config_is_built_from_trainer_hyperparameters(self, tmp_path):
        mrt = _mrt()
        mrt.model_name = "org/base-model"
        mrt._trainer = types.SimpleNamespace(lora_r=8, lora_alpha=24)  # no live model

        mrt._write_eval_adapter_config(tmp_path)

        cfg = json.loads((tmp_path / "adapter_config.json").read_text())
        assert cfg["r"] == 8 and cfg["lora_alpha"] == 24
        assert cfg["base_model_name_or_path"] == "org/base-model"
        assert cfg["task_type"] == "CAUSAL_LM"

    def test_library_defaults_when_nothing_is_recorded(self, tmp_path):
        mrt = _mrt()
        mrt._trainer = None
        mrt._write_eval_adapter_config(tmp_path)
        cfg = json.loads((tmp_path / "adapter_config.json").read_text())
        assert cfg["r"] == 16 and cfg["lora_alpha"] == 32


class TestLoadEvalReferences:
    def test_missing_file(self, tmp_path):
        with pytest.raises(BackpropagateError) as exc_info:
            _mrt()._load_eval_references(str(tmp_path / "none.jsonl"))
        assert exc_info.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"
        assert "not found" in str(exc_info.value)

    def test_blank_lines_bad_json_and_non_objects_are_skipped(self, tmp_path):
        path = tmp_path / "refs.jsonl"
        path.write_text(
            '{"prompt": "p1", "reference": "r1"}\n'
            "\n"
            "this is not json\n"
            '["a", "list"]\n'
            '{"prompt": "p2", "references": ["r2a", "r2b"]}\n',
            encoding="utf-8",
        )

        items = _mrt()._load_eval_references(str(path))

        assert items == [
            {"prompt": "p1", "reference": "r1"},
            {"prompt": "p2", "references": ["r2a", "r2b"]},
        ]

    def test_file_with_no_usable_rows(self, tmp_path):
        path = tmp_path / "refs.jsonl"
        path.write_text("nope\n\n[1]\n", encoding="utf-8")
        with pytest.raises(BackpropagateError) as exc_info:
            _mrt()._load_eval_references(str(path))
        assert "produced no usable rows" in str(exc_info.value)


# =============================================================================
# _evaluate_accumulator
# =============================================================================


class _EvalHarness:
    """A trainer with a real checkpoint manager/history and ``evaluate_run`` recorded."""

    def __init__(self, monkeypatch, tmp_path, **cfg):
        self.tmp_path = tmp_path
        self.mrt = _mrt(checkpoint_dir=str(tmp_path), **cfg)
        self.mrt._run_id = "sess"
        self.mrt._trainer = FakeInnerTrainer(build_peft_llama())
        self.mrt._checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(tmp_path), policy=CheckpointPolicy()
        )
        self.calls = []
        self.during = {}
        self.tmp_roots = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(*a, **k):
            path = real_mkdtemp(*a, **k)
            self.tmp_roots.append(Path(path))
            return path

        monkeypatch.setattr("tempfile.mkdtemp", tracking_mkdtemp)

        def fake_evaluate_run(run_id, **kwargs):
            self.calls.append((run_id, kwargs))
            entry = RunHistoryManager(str(tmp_path)).get_run(run_id)
            self.during["entry"] = entry
            if entry:
                adapter = Path(entry["checkpoint_path"])
                self.during["files"] = sorted(p.name for p in adapter.iterdir())
            return EvalResult(run_id=run_id, model_name="m", held_out_loss=1.0, perplexity=2.7)

        monkeypatch.setattr(multi_run, "evaluate_run", fake_evaluate_run)


class TestEvaluateAccumulator:
    def test_derived_holdout_is_passed_and_transient_artifacts_are_cleaned_up(
        self, monkeypatch, tmp_path
    ):
        h = _EvalHarness(monkeypatch, tmp_path)
        accumulator = {"x.lora_B.w": torch.ones(2, 2)}
        ds = Dataset.from_dict({"text": [f"row{i}" for i in range(20)]})

        result = h.mrt._evaluate_accumulator(accumulator, 3, ds, phase="after")

        assert result.held_out_loss == 1.0
        (eval_run_id, kwargs), = h.calls
        assert eval_run_id == "sess-evalgate-after-003"
        assert kwargs["output_dir"] == str(tmp_path)
        assert kwargs["heldout"] is None
        assert kwargs["heldout_texts"] == ["row18", "row19"]  # reserved last decile
        assert kwargs["metrics"] is None and kwargs["references"] is None
        # while evaluate_run ran: a transient entry pointed at a real adapter dir
        entry = h.during["entry"]
        assert entry["session_kind"] == "eval_gate_transient"
        assert h.during["files"] == ["adapter_config.json", "adapter_model.bin"]
        # ... and afterwards both the entry and the temp directory are gone
        assert RunHistoryManager(str(tmp_path)).get_run(eval_run_id) is None
        assert all(not root.exists() for root in h.tmp_roots)

    def test_the_accumulator_weights_are_what_gets_saved(self, monkeypatch, tmp_path):
        h = _EvalHarness(monkeypatch, tmp_path)
        saved = {}
        real_save = torch.save

        def spy_save(obj, path, *a, **k):
            saved["state"], saved["name"] = obj, Path(path).name
            return real_save(obj, path, *a, **k)

        monkeypatch.setattr(torch, "save", spy_save)
        accumulator = {"x.lora_B.w": torch.arange(4.0)}

        h.mrt._evaluate_accumulator(accumulator, 1, text_dataset(20), phase="before")

        assert saved["name"] == "adapter_model.bin"
        assert saved["state"] is accumulator

    def test_explicit_heldout_path_wins_over_deriving_one(self, monkeypatch, tmp_path):
        h = _EvalHarness(monkeypatch, tmp_path, eval_heldout_path="held.jsonl")

        h.mrt._evaluate_accumulator({"x.lora_B.w": torch.ones(1)}, 2, None, phase="before")

        (_, kwargs), = h.calls
        assert kwargs["heldout"] == "held.jsonl" and kwargs["heldout_texts"] is None

    def test_no_accumulator_means_no_files_but_still_an_evaluation(self, monkeypatch, tmp_path):
        h = _EvalHarness(monkeypatch, tmp_path)

        h.mrt._evaluate_accumulator(None, 1, text_dataset(20), phase="before")

        assert h.during["files"] == []
        assert len(h.calls) == 1

    def test_task_metrics_and_references_are_forwarded_together(self, monkeypatch, tmp_path):
        refs = tmp_path / "refs.jsonl"
        refs.write_text('{"prompt": "p", "reference": "r"}\n', encoding="utf-8")
        h = _EvalHarness(
            monkeypatch, tmp_path, eval_metrics=["exact_match"], eval_references_path=str(refs)
        )

        h.mrt._evaluate_accumulator({"x.lora_B.w": torch.ones(1)}, 1, text_dataset(20),
                                    phase="after")

        (_, kwargs), = h.calls
        assert kwargs["metrics"] == ["exact_match"]
        assert kwargs["references"] == [{"prompt": "p", "reference": "r"}]

    def test_references_are_ignored_without_a_metric_selection(self, monkeypatch, tmp_path):
        h = _EvalHarness(monkeypatch, tmp_path, eval_references_path="/never/read.jsonl")

        h.mrt._evaluate_accumulator({"x.lora_B.w": torch.ones(1)}, 1, text_dataset(20),
                                    phase="after")

        (_, kwargs), = h.calls
        assert kwargs["references"] is None  # and the unreadable path was never opened

    def test_transient_history_failure_does_not_block_the_evaluation(
        self, monkeypatch, tmp_path, caplog
    ):
        h = _EvalHarness(monkeypatch, tmp_path)

        def boom(self, **kwargs):
            raise OSError("history locked")

        monkeypatch.setattr(RunHistoryManager, "record_run_started", boom)

        with caplog.at_level(logging.DEBUG, logger=MR_LOGGER):
            h.mrt._evaluate_accumulator({"x.lora_B.w": torch.ones(1)}, 1, text_dataset(20),
                                        phase="after")

        assert len(h.calls) == 1
        assert any("transient history write failed (non-fatal)" in r.getMessage()
                   for r in caplog.records)

    def test_cleanup_failures_are_swallowed_and_the_result_is_returned(
        self, monkeypatch, tmp_path
    ):
        h = _EvalHarness(monkeypatch, tmp_path)

        def boom(self, run_id):
            raise OSError("cannot delete")

        monkeypatch.setattr(RunHistoryManager, "delete_run", boom)

        result = h.mrt._evaluate_accumulator({"x.lora_B.w": torch.ones(1)}, 1, text_dataset(20),
                                             phase="after")

        assert result.held_out_loss == 1.0
        assert all(not root.exists() for root in h.tmp_roots)  # tmp dir still removed

    def test_unresolvable_holdout_aborts_but_still_cleans_up(self, monkeypatch, tmp_path):
        h = _EvalHarness(monkeypatch, tmp_path)

        with pytest.raises(BackpropagateError) as exc_info:
            h.mrt._evaluate_accumulator({"x.lora_B.w": torch.ones(1)}, 1, None, phase="before")

        assert exc_info.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"
        assert h.calls == []
        assert all(not root.exists() for root in h.tmp_roots)
