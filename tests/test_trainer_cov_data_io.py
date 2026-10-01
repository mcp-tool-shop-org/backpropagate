"""Coverage tests for the data / persistence side of ``Trainer``: dataset
loading + reasoning-trace filtering, pre-tokenization, KTO weighting, the
offload run wrapper, atomic ``save``, ``export`` routing, ``multi_run``
delegation and the module-level convenience functions.

Real objects wherever possible: real ``DatasetLoader`` / ``datasets.Dataset``
on tmp files, a real tiny Llama for ``save`` (``save_pretrained`` runs for
real, the promote/backup dance runs on the real filesystem), a real tiny
tokenizer for the trace filter and ``_pre_tokenize``.

Mock boundary (named per test): the HF Hub / network (``datasets.load_dataset``
for dataset *names*), CUDA/NCCL (the FSDP offload engine and FSDP state-dict
gather), llama.cpp / HF export tooling (``backpropagate.export.export_*``),
the multi-run trainer (own module, own tests) and OS failure injection
(``shutil.move`` / ``os.rename`` / ``Path.mkdir`` raising).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")

from backpropagate import trainer as T  # noqa: E402
from backpropagate.checkpoints import CheckpointManager, RunHistoryManager  # noqa: E402
from backpropagate.datasets import (  # noqa: E402
    DatasetFormat,
    DatasetLoader,
    ValidationError,
    ValidationResult,
)
from backpropagate.exceptions import (  # noqa: E402
    BackpropagateError,
    CheckpointError,
    DatasetError,
    DatasetFormatError,
    DatasetNotFoundError,
    GPUNotAvailableError,
    InvalidSettingError,
    TrainingError,
)
from backpropagate.trainer import Trainer, TrainingCallback  # noqa: E402
from tests.helpers.tiny_models import tiny_llama, tiny_tokenizer  # noqa: E402

LOGGER = "backpropagate.trainer"
Dataset = datasets.Dataset


def make(tmp_path, **kw):
    base = {
        "model": "acme/Tiny-1B", "use_unsloth": False, "batch_size": 2, "max_seq_length": 32,
        "output_dir": str(tmp_path / "out"), "report_to": "none",
    }
    base.update(kw)
    return Trainer(**base)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(T, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(T, "_RETRY_MAX_SECONDS", 0)


CHAT_ROW = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
PREF_ROW = {"prompt": "q", "chosen": "good", "rejected": "bad"}
KTO_ROWS = [{"prompt": "q", "completion": "a", "label": True},
            {"prompt": "q", "completion": "b", "label": False}]


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def failing_validation():
    errs = [ValidationError(i, "format", "unknown_format", f"bad row {i}") for i in range(7)]
    warns = [ValidationError(i, "length", "short", f"short row {i}") for i in range(7)]
    return ValidationResult(is_valid=False, total_rows=10, valid_rows=3, errors=errs,
                            warnings=warns, format_detected=DatasetFormat.OPENAI)


# ---------------------------------------------------------------------------
# _load_dataset
# ---------------------------------------------------------------------------

class TestLoadDatasetDefaultAndHub:
    def test_default_dataset_for_preference_methods_must_carry_pairs(self, tmp_path):
        text_only = Dataset.from_dict({"text": ["a", "b"]})
        with patch("datasets.load_dataset", return_value=text_only):
            t = make(tmp_path, method="orpo", learning_rate=1e-5)
            with pytest.raises(DatasetFormatError) as ei:
                t._load_dataset(None, method="orpo")
        assert ei.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"
        assert "default HuggingFace dataset" in str(ei.value)

    def test_default_dataset_for_kto_must_carry_completion_and_label(self, tmp_path):
        with patch("datasets.load_dataset", return_value=Dataset.from_dict({"text": ["a"]})):
            t = make(tmp_path, method="kto", learning_rate=1e-6)
            with pytest.raises(DatasetFormatError, match="completion/label"):
                t._load_dataset(None, method="kto")

    def test_default_dataset_sft_is_returned_as_is(self, tmp_path):
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})
        with patch("datasets.load_dataset", return_value=ds):
            out = make(tmp_path)._load_dataset(None, samples=0)
        assert list(out["text"]) == ["a", "b", "c"]

    def test_default_dataset_file_not_found_is_structured(self, tmp_path):
        with patch("datasets.load_dataset", side_effect=FileNotFoundError("cache dir missing")):
            with pytest.raises(DatasetNotFoundError):
                make(tmp_path)._load_dataset(None)

    def test_default_dataset_unexpected_failure_is_wrapped(self, tmp_path):
        with patch("datasets.load_dataset", side_effect=ValueError("arrow exploded")):
            with pytest.raises(DatasetError, match="Failed to load dataset: arrow exploded"):
                make(tmp_path)._load_dataset(None)

    def test_hub_name_success_with_pair_validation(self, tmp_path):
        pairs = Dataset.from_list([PREF_ROW])
        with patch("datasets.load_dataset", return_value=pairs) as m:
            out = make(tmp_path, method="orpo", learning_rate=1e-5)._load_dataset(
                "org/prefs", method="orpo")
        assert out.column_names == ["prompt", "chosen", "rejected"]
        assert m.call_args.args[0] == "org/prefs"

    @pytest.mark.parametrize("method,bad_cols", [("simpo", {"text": ["a"]}), ("kto", {"chosen": ["a"]})])
    def test_hub_name_with_wrong_columns_is_refused(self, tmp_path, method, bad_cols):
        with patch("datasets.load_dataset", return_value=Dataset.from_dict(bad_cols)):
            t = make(tmp_path, method=method, learning_rate=1e-6)
            with pytest.raises(DatasetFormatError) as ei:
                t._load_dataset("org/data", method=method)
        assert "HuggingFace dataset 'org/data'" in str(ei.value)

    def test_hub_name_load_failure_is_a_dataset_error_with_hint(self, tmp_path):
        with patch("datasets.load_dataset", side_effect=ValueError("gone")):
            with pytest.raises(DatasetError) as ei:
                make(tmp_path)._load_dataset("org/missing")
        assert "Failed to load HuggingFace dataset 'org/missing': gone" in str(ei.value)
        assert "network connection" in ei.value.suggestion

    def test_unsupported_dataset_type_rejected(self, tmp_path):
        with pytest.raises(DatasetError, match="Unsupported dataset type: int"):
            make(tmp_path)._load_dataset(42)


class TestLoadDatasetLoaders:
    def test_loader_validation_warnings_and_errors_are_logged_but_not_fatal(self, tmp_path, caplog):
        loader = DatasetLoader([CHAT_ROW])
        with patch.object(DatasetLoader, "validation_result", new_callable=PropertyMock,
                          return_value=failing_validation()), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            out = make(tmp_path)._load_dataset(loader, samples=0)
        assert len(out) == 1
        assert caplog.text.count("Dataset validation warning:") == 5  # capped at 5
        assert "Dataset has 7 validation errors (70.0% error rate) — proceeding anyway" in caplog.text
        assert caplog.text.count("  Row ") == 5

    def test_loader_routes_to_preference_and_kto_views(self, tmp_path):
        pref = make(tmp_path, method="orpo", learning_rate=1e-5)._load_dataset(
            DatasetLoader([PREF_ROW]), samples=0, method="orpo")
        assert {"chosen", "rejected"} <= set(pref.column_names)
        kto = make(tmp_path, method="kto", learning_rate=1e-6)._load_dataset(
            DatasetLoader(KTO_ROWS), samples=0, method="kto")
        assert {"completion", "label"} <= set(kto.column_names) and len(kto) == 2

    def test_sft_on_preference_loader_warns_and_trains_on_chosen(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = make(tmp_path)._load_dataset(DatasetLoader([PREF_ROW]), samples=0)
        assert "Dataset looks like preference pairs" in caplog.text
        assert len(out) == 1 and "chosen" not in out.column_names

    def test_sft_on_plain_loader_is_silent(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            make(tmp_path)._load_dataset(DatasetLoader([CHAT_ROW]), samples=0)
        assert "preference pairs" not in caplog.text


class TestLoadDatasetFiles:
    def test_missing_file_is_dataset_not_found(self, tmp_path):
        with pytest.raises(DatasetNotFoundError) as ei:
            make(tmp_path)._load_dataset(str(tmp_path / "nope.jsonl"))
        assert "nope.jsonl" in str(ei.value)
        assert "Check the file path" in ei.value.suggestion

    def test_file_vanishing_between_exists_check_and_read_is_not_found(self, tmp_path):
        f = write_jsonl(tmp_path / "gone.jsonl", [CHAT_ROW])
        with patch.object(DatasetLoader, "__init__", side_effect=FileNotFoundError("vanished")):
            with pytest.raises(DatasetNotFoundError) as ei:
                make(tmp_path)._load_dataset(f)
        assert "Create the file or use a HuggingFace dataset name" in ei.value.suggestion

    def test_file_validation_noise_is_logged(self, tmp_path, caplog):
        f = write_jsonl(tmp_path / "d.jsonl", [CHAT_ROW])
        with patch.object(DatasetLoader, "validation_result", new_callable=PropertyMock,
                          return_value=failing_validation()), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            out = make(tmp_path)._load_dataset(f, samples=0)
        assert len(out) == 1
        assert "validation errors (70.0% error rate)" in caplog.text
        assert caplog.text.count("Dataset validation warning:") == 5

    def test_kto_file_routes_through_unpaired_view(self, tmp_path):
        f = write_jsonl(tmp_path / "k.jsonl", KTO_ROWS)
        out = make(tmp_path, method="kto", learning_rate=1e-6)._load_dataset(f, samples=0, method="kto")
        assert sorted(out["label"]) == [False, True]

    def test_pair_file_routes_through_preference_view(self, tmp_path):
        f = write_jsonl(tmp_path / "p.jsonl", [PREF_ROW])
        out = make(tmp_path, method="orpo", learning_rate=1e-5)._load_dataset(f, samples=0, method="orpo")
        assert out[0]["chosen"] == "good"

    def test_kto_on_sft_file_is_a_format_error(self, tmp_path):
        f = write_jsonl(tmp_path / "s.jsonl", [CHAT_ROW])
        t = make(tmp_path, method="kto", learning_rate=1e-6)
        with pytest.raises(DatasetFormatError):
            t._load_dataset(f, samples=0, method="kto")


class TestSampleCapAndShuffle:
    def _ds(self, n=10):
        return Dataset.from_dict({"text": [f"row{i}" for i in range(n)]})

    def test_cap_without_shuffle_keeps_the_head(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(T.settings.data, "shuffle", False)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            out = make(tmp_path)._load_dataset(self._ds(), samples=3)
        assert list(out["text"]) == ["row0", "row1", "row2"]
        assert "Using 3 of 10 dataset rows" in caplog.text

    def test_cap_with_shuffle_is_seeded_and_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T.settings.data, "shuffle", True)
        a = list(make(tmp_path)._load_dataset(self._ds(), samples=4)["text"])
        b = list(make(tmp_path)._load_dataset(self._ds(), samples=4)["text"])
        assert a == b and len(set(a)) == 4

    def test_zero_samples_disables_the_cap(self, tmp_path):
        assert len(make(tmp_path)._load_dataset(self._ds(), samples=0)) == 10


# ---------------------------------------------------------------------------
# Reasoning-trace filter
# ---------------------------------------------------------------------------

def think(n_words):
    return "<think>" + " ".join(["the cat sat"] * max(1, n_words // 3)) + "</think> the answer"


class TestReasoningTraceFilter:
    def _trainer(self, tmp_path, **kw):
        t = make(tmp_path, reasoning_trace=True, **kw)
        t._tokenizer = tiny_tokenizer()
        return t

    def test_flag_filters_rows_without_think_spans(self, tmp_path):
        ds = Dataset.from_dict({"text": [think(9), "no trace here", think(12)]})
        out = self._trainer(tmp_path)._load_dataset(ds, samples=0)
        assert all("<think>" in x for x in out["text"]) and len(out) == 2

    def test_missing_text_column_skips_with_warning(self, tmp_path, caplog):
        ds = Dataset.from_dict({"other": ["a"]})
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = self._trainer(tmp_path)._filter_reasoning_traces(ds)
        assert out is ds and "no 'text' column" in caplog.text

    def test_missing_tokenizer_skips_with_warning(self, tmp_path, caplog):
        t = make(tmp_path, reasoning_trace=True)
        ds = Dataset.from_dict({"text": [think(9)]})
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert t._filter_reasoning_traces(ds) is ds
        assert "tokenizer not loaded" in caplog.text

    def test_encode_failure_falls_back_to_whitespace_count(self, tmp_path):
        t = self._trainer(tmp_path)

        class Broken:
            def encode(self, text):
                raise RuntimeError("tokenizer crashed")

            def apply_chat_template(self, *a, **k):
                raise RuntimeError("no template")

        t._tokenizer = Broken()
        ds = Dataset.from_dict({"text": [think(9), "no trace"]})
        out = t._filter_reasoning_traces(ds)
        assert list(out["text"]) == [think(9)]

    def test_template_that_injects_think_is_probed(self, tmp_path):
        tok = tiny_tokenizer()
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {% if m['role'] == 'assistant' %}"
            "<think></think>{% endif %}{{ m['content'] }}\n{% endfor %}"
        )
        t = self._trainer(tmp_path)
        t._tokenizer = tok
        ds = Dataset.from_dict({"text": [think(9)]})
        assert len(t._filter_reasoning_traces(ds)) == 1

    def test_filter_failure_returns_unfiltered_dataset(self, tmp_path, caplog):
        t = self._trainer(tmp_path)
        ds = Dataset.from_dict({"text": [think(9)]})
        with patch("backpropagate.datasets.filter_by_trace_length", side_effect=ValueError("bad bounds")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            assert t._filter_reasoning_traces(ds) is ds
        assert "trace-length filtering failed" in caplog.text

    def test_everything_filtered_returns_original_with_warning(self, tmp_path, caplog):
        t = self._trainer(tmp_path)
        ds = Dataset.from_dict({"text": ["plain one", "plain two"]})
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert t._filter_reasoning_traces(ds) is ds
        assert "removed every sample" in caplog.text

    def test_custom_text_column_is_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T.settings.data, "text_column", "body")
        t = self._trainer(tmp_path)
        ds = Dataset.from_dict({"body": [think(9), "no trace"]})
        out = t._filter_reasoning_traces(ds)
        assert out.column_names == ["body"] and len(out) == 1


# ---------------------------------------------------------------------------
# _pre_tokenize
# ---------------------------------------------------------------------------

class TestPreTokenize:
    def _t(self, tmp_path):
        t = make(tmp_path, max_seq_length=8)
        t._tokenizer = tiny_tokenizer()
        return t

    def test_tokenizes_and_truncates_text_column(self, tmp_path):
        ds = Dataset.from_dict({"text": ["the cat sat on the mat and ran to the park", "dog"]})
        out = self._t(tmp_path)._pre_tokenize(ds)
        assert set(out.column_names) >= {"input_ids", "attention_mask"} and "text" not in out.column_names
        assert [len(x) for x in out["input_ids"]] == [8, 1]

    def test_missing_text_column_is_a_dataset_error(self, tmp_path):
        with pytest.raises(DatasetError):
            self._t(tmp_path)._pre_tokenize(Dataset.from_dict({"body": ["x"]}))

    def test_tokenizer_failure_is_wrapped(self, tmp_path):
        t = self._t(tmp_path)

        def boom(*a, **k):
            raise RuntimeError("fast tokenizer panicked")

        t._tokenizer = boom
        with pytest.raises(DatasetError, match="Tokenization failed: fast tokenizer panicked"):
            t._pre_tokenize(Dataset.from_dict({"text": ["x"]}))


# ---------------------------------------------------------------------------
# KTO auto-weighting corner
# ---------------------------------------------------------------------------

class TestKtoWeightWarning:
    def test_operator_weights_outside_band_warn_and_get_rebalanced(self, tmp_path, caplog):
        t = make(tmp_path, method="kto", learning_rate=1e-6,
                 kto_desirable_weight=3.0, kto_undesirable_weight=1.0)
        ds = Dataset.from_list([{"prompt": "q", "completion": "a", "label": True},
                                {"prompt": "q", "completion": "b", "label": True},
                                {"prompt": "q", "completion": "c", "label": False},
                                {"prompt": "q", "completion": "d", "label": False}])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            t._auto_balance_kto_weights(ds)
        assert "your explicit weights (desirable=3.000, undesirable=1.000)" in caplog.text
        assert t._kto_resolved_desirable_weight == pytest.approx(3.0)
        assert t._kto_resolved_undesirable_weight == pytest.approx(3.0 / (4 / 3))  # 2.25
        ratio = (t._kto_resolved_desirable_weight * 2) / (t._kto_resolved_undesirable_weight * 2)
        assert ratio == pytest.approx(4 / 3)


# ---------------------------------------------------------------------------
# Offload run wrapper (engine = CUDA/NCCL boundary, patched)
# ---------------------------------------------------------------------------

@pytest.fixture
def offload_trainer(tmp_path, monkeypatch):
    from backpropagate import offload_engine as oe

    monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: (256.0, 200.0))
    monkeypatch.setattr(T, "_detect_total_vram_gb", lambda: 32.0)
    t = make(tmp_path, mode="full", full_ft_offload=True, learning_rate=1e-5)
    t._model, t._tokenizer = object(), object()
    return t


def engine_result(**over):
    r = {"model": "TRAINED", "optimizer": "OPT", "losses": [3.0, float("nan"), 1.5],
         "samples_seen": 12, "step_times": [0.1, 0.2, 0.3], "update_retention": [0.9]}
    r.update(over)
    return r


class TestTrainFullOffload:
    def test_success_records_history_metadata_and_fires_callbacks(self, offload_trainer, tmp_path):
        seen = {}

        def fake_engine(model, tok, ds, **kw):
            seen.update(kw)
            seen["on_step_cb"] = kw["on_step"]
            return engine_result()

        done = []
        cb = TrainingCallback(on_step=lambda s, loss: None, on_complete=done.append)
        with patch("backpropagate.offload_engine.run_offload_training", side_effect=fake_engine):
            run = offload_trainer._train_full_offload("DS", dataset="d.jsonl", steps=40, callback=cb)
        assert run.steps == 40 and run.final_loss == 1.5
        assert run.loss_history == [3.0, 1.5]  # NaN dropped
        assert run.samples_seen == 12 and run.output_path == str(offload_trainer.output_dir)
        assert run.metadata["engine"] == "fsdp2-direct"
        assert run.metadata["update_retention"] == [0.9]
        assert offload_trainer._model == "TRAINED" and offload_trainer._offload_optimizer == "OPT"
        assert offload_trainer._has_trained is True and done == [run]
        assert seen["steps"] == 40 and seen["batch_size"] == 2
        assert seen["on_step_cb"] is cb.on_step
        assert seen["warmup_steps"] == min(T.settings.training.warmup_steps, 4)
        (row,) = RunHistoryManager(str(tmp_path / "out")).get_history()
        assert row["status"] == "completed" and row["hyperparameters"]["full_ft_offload"] is True

    def test_empty_loss_list_records_zero(self, offload_trainer):
        with patch("backpropagate.offload_engine.run_offload_training",
                   return_value=engine_result(losses=[])):
            run = offload_trainer._train_full_offload("DS", dataset=None, steps=None, callback=None)
        assert run.final_loss == 0.0 and run.loss_history == []

    def test_history_failures_and_callback_failures_are_isolated(self, offload_trainer, caplog):
        def bad(run):
            raise RuntimeError("cb boom")

        with patch("backpropagate.offload_engine.run_offload_training", return_value=engine_result()), \
                patch.object(RunHistoryManager, "record_run_started", side_effect=OSError("h1")), \
                patch.object(RunHistoryManager, "record_run_completed", side_effect=OSError("h2")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            run = offload_trainer._train_full_offload(
                "DS", dataset="d", steps=3, callback=TrainingCallback(on_complete=bad))
        assert run.steps == 3
        assert "record_run_started failed: h1" in caplog.text
        assert "record_run_completed failed: h2" in caplog.text
        assert "on_complete callback raised error: cb boom" in caplog.text

    def test_engine_failure_is_wrapped_and_reported(self, offload_trainer, tmp_path, caplog):
        errors = []

        def bad_cb(exc):
            errors.append(exc)
            raise RuntimeError("cb exploded too")

        with patch("backpropagate.offload_engine.run_offload_training",
                   side_effect=RuntimeError("NCCL timeout")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(TrainingError, match="full_ft_offload training failed: NCCL timeout") as ei:
                offload_trainer._train_full_offload(
                    "DS", dataset="d", steps=3, callback=TrainingCallback(on_error=bad_cb))
        assert isinstance(ei.value.__cause__, RuntimeError) and len(errors) == 1
        assert "on_error callback raised error: cb exploded too" in caplog.text
        (row,) = RunHistoryManager(str(tmp_path / "out")).get_history()
        assert row["status"] == "failed" and "NCCL timeout" in row["failure_reason"]

    def test_structured_engine_errors_pass_through_even_if_history_is_broken(self, offload_trainer):
        original = TrainingError("fit check failed", code="RUNTIME_TRAINING_FAILED")
        with patch("backpropagate.offload_engine.run_offload_training", side_effect=original), \
                patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro")):
            with pytest.raises(BackpropagateError) as ei:
                offload_trainer._train_full_offload("DS", dataset="d", steps=3, callback=None)
        assert ei.value is original

    def test_non_integer_batch_defaults_to_one(self, offload_trainer):
        offload_trainer.batch_size = "auto"
        seen = {}
        with patch("backpropagate.offload_engine.run_offload_training",
                   side_effect=lambda *a, **k: seen.update(k) or engine_result()):
            offload_trainer._train_full_offload("DS", dataset="d", steps=3, callback=None)
        assert seen["batch_size"] == 1


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

@pytest.fixture
def saver(tmp_path):
    """Trainer holding a real tiny full-precision model + tokenizer."""
    t = make(tmp_path, mode="full", learning_rate=1e-5)
    t._model = tiny_llama(layers=1)
    t._tokenizer = tiny_tokenizer()
    t._is_loaded = True
    t._has_trained = True
    return t


def saved_ok(path):
    p = Path(path)
    return (p / "config.json").exists() and any(p.glob("*.safetensors")) and (p / "tokenizer.json").exists()


class TestSave:
    def test_save_writes_model_registers_manifest_and_returns_path(self, saver, tmp_path):
        dest = tmp_path / "ckpt" / "final"
        out = saver.save(str(dest), run_id="run-xyz")
        assert out == str(dest) and saved_ok(dest)
        assert (dest / "run_id").read_text() == "run-xyz"
        assert not (tmp_path / "ckpt" / "final.partial").exists()
        cm = CheckpointManager(str(saver.output_dir))
        found = cm.find_latest_for_run_id("run-xyz")
        assert found is not None and found.path == str(dest) and found.is_run_boundary

    def test_untrained_save_warns(self, saver, tmp_path, caplog):
        saver._has_trained = False
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            saver.save(str(tmp_path / "x"), register_in_manifest=False)
        assert "Saving a loaded but untrained model" in caplog.text
        assert not (Path(saver.output_dir) / "manifest.json").exists()

    def test_parent_path_that_is_a_file_is_a_checkpoint_error(self, saver, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        with pytest.raises(CheckpointError) as ei:
            saver.save(str(blocker / "child"))
        assert "Failed to create parent directory" in str(ei.value)

    def test_stale_partial_dir_is_wiped(self, saver, tmp_path):
        dest = tmp_path / "ckpt"
        stale = tmp_path / "ckpt.partial"
        stale.mkdir()
        (stale / "garbage.bin").write_bytes(b"x")
        saver.save(str(dest), register_in_manifest=False)
        assert saved_ok(dest) and not stale.exists()

    def test_crashed_prior_save_is_recovered_then_replaced_without_losing_it(
        self, saver, tmp_path, caplog
    ):
        dest = tmp_path / "ckpt"
        backup = tmp_path / "ckpt.backup"
        backup.mkdir()
        (backup / "prior.txt").write_text("prior checkpoint")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            saver.save(str(dest), register_in_manifest=False)
        assert "Recovering checkpoint from a previous crashed save" in caplog.text
        assert saved_ok(dest) and not backup.exists()

    def test_failed_backup_recovery_is_only_a_warning(self, saver, tmp_path, caplog):
        dest = tmp_path / "ckpt"
        backup = tmp_path / "ckpt.backup"
        backup.mkdir()
        real_move = shutil.move

        def flaky_move(src, dst, *a, **k):
            if str(src).endswith(".backup"):
                raise OSError("cross-device link")
            return real_move(src, dst, *a, **k)

        with patch("shutil.move", side_effect=flaky_move), caplog.at_level(logging.WARNING, logger=LOGGER):
            saver.save(str(dest), register_in_manifest=False)
        assert "Failed to recover stale backup checkpoint: cross-device link" in caplog.text
        assert saved_ok(dest)

    def test_stale_backup_next_to_a_live_checkpoint_is_discarded(self, saver, tmp_path):
        dest = tmp_path / "ckpt"
        saver.save(str(dest), register_in_manifest=False)
        backup = tmp_path / "ckpt.backup"
        backup.mkdir()
        (backup / "old.txt").write_text("x")
        saver.save(str(dest), register_in_manifest=False)
        assert saved_ok(dest) and not backup.exists()

    def test_overwrite_replaces_prior_checkpoint(self, saver, tmp_path):
        dest = tmp_path / "ckpt"
        saver.save(str(dest), run_id="one", register_in_manifest=False)
        saver.save(str(dest), run_id="two", register_in_manifest=False)
        assert (dest / "run_id").read_text() == "two"
        assert not (tmp_path / "ckpt.backup").exists() and not (tmp_path / "ckpt.partial").exists()

    def test_partial_dir_creation_failure_is_a_checkpoint_error(self, saver, tmp_path):
        real_mkdir = Path.mkdir

        def deny(self, *a, **k):
            if self.name.endswith(".partial"):
                raise OSError("read-only filesystem")
            return real_mkdir(self, *a, **k)

        with patch.object(Path, "mkdir", deny):
            with pytest.raises(CheckpointError, match="Failed to create partial directory"):
                saver.save(str(tmp_path / "ckpt"))

    def test_fsdp_sharded_model_saves_the_gathered_state_dict(self, saver, tmp_path):
        gathered = {k: v.clone() for k, v in saver._model.state_dict().items()}
        calls = {}
        real_save = saver._model.save_pretrained

        def spy(path, **kw):
            calls.update(kw)
            return real_save(path, **kw)

        saver._model.save_pretrained = spy
        with patch.object(T, "_gather_fsdp_full_state_dict", return_value=gathered):
            saver.save(str(tmp_path / "ckpt"), register_in_manifest=False)
        assert calls["state_dict"] is gathered
        assert saved_ok(tmp_path / "ckpt")

    def test_checkpoint_errors_raised_while_writing_propagate_unwrapped(self, saver, tmp_path):
        boom = CheckpointError("save", str(tmp_path / "ckpt"), "gather failed")
        with patch.object(T, "_gather_fsdp_full_state_dict", side_effect=boom):
            with pytest.raises(CheckpointError) as ei:
                saver.save(str(tmp_path / "ckpt"))
        assert ei.value is boom
        assert not (tmp_path / "ckpt.partial").exists()  # cleaned up in finally

    def test_other_write_failures_are_wrapped_in_checkpoint_error(self, saver, tmp_path):
        with patch.object(T, "_gather_fsdp_full_state_dict", side_effect=MemoryError("host RAM")):
            with pytest.raises(CheckpointError) as ei:
                saver.save(str(tmp_path / "ckpt"))
        assert "host RAM" in str(ei.value)

    def test_run_id_file_write_failure_is_a_warning(self, saver, tmp_path, caplog):
        real_write = Path.write_text

        def deny(self, *a, **k):
            if self.name == "run_id":
                raise OSError("disk quota")
            return real_write(self, *a, **k)

        with patch.object(Path, "write_text", deny), caplog.at_level(logging.WARNING, logger=LOGGER):
            saver.save(str(tmp_path / "ckpt"), run_id="r1", register_in_manifest=False)
        assert "Failed to write run_id file: disk quota" in caplog.text
        assert saved_ok(tmp_path / "ckpt")

    def test_backup_appearing_mid_save_is_cleared_before_the_rename(self, saver, tmp_path):
        dest = tmp_path / "ckpt"
        saver.save(str(dest), register_in_manifest=False)
        real_save = saver._model.save_pretrained

        def create_backup_then_save(path, **kw):
            (tmp_path / "ckpt.backup").mkdir()
            (tmp_path / "ckpt.backup" / "late.txt").write_text("x")
            return real_save(path, **kw)

        saver._model.save_pretrained = create_backup_then_save
        saver.save(str(dest), register_in_manifest=False)
        assert saved_ok(dest) and not (tmp_path / "ckpt.backup").exists()

    def test_promote_failure_restores_the_prior_checkpoint(self, saver, tmp_path, caplog):
        dest = tmp_path / "ckpt"
        saver.save(str(dest), run_id="prior", register_in_manifest=False)
        real_move = shutil.move

        def fail_promote(src, dst, *a, **k):
            if str(src).endswith(".partial"):
                raise OSError("disk full during promote")
            return real_move(src, dst, *a, **k)

        with patch("shutil.move", side_effect=fail_promote), caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(CheckpointError, match="disk full during promote"):
                saver.save(str(dest), run_id="new", register_in_manifest=False)
        assert (dest / "run_id").read_text() == "prior"  # prior checkpoint restored intact
        assert "Promote failed; restored the prior checkpoint" in caplog.text
        assert not (tmp_path / "ckpt.partial").exists()

    def test_promote_failure_with_failed_restore_names_where_the_prior_lives(
        self, saver, tmp_path, caplog
    ):
        dest = tmp_path / "ckpt"
        saver.save(str(dest), run_id="prior", register_in_manifest=False)
        real_move, real_rename = shutil.move, os.rename

        def fail_promote(src, dst, *a, **k):
            if str(src).endswith(".partial"):
                raise OSError("promote failed")
            return real_move(src, dst, *a, **k)

        def fail_restore(src, dst, *a, **k):
            if str(src).endswith(".backup"):
                raise OSError("restore failed")
            return real_rename(src, dst, *a, **k)

        with patch("shutil.move", side_effect=fail_promote), \
                patch("os.rename", side_effect=fail_restore), \
                caplog.at_level(logging.ERROR, logger=LOGGER):
            with pytest.raises(CheckpointError):
                saver.save(str(dest), register_in_manifest=False)
        assert "Promote failed AND prior-checkpoint restore failed: restore failed" in caplog.text
        assert (tmp_path / "ckpt.backup" / "run_id").read_text() == "prior"  # still recoverable

    def test_manifest_registration_failure_never_breaks_the_save(self, saver, tmp_path, caplog):
        with patch.object(CheckpointManager, "register", side_effect=OSError("manifest locked")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            out = saver.save(str(tmp_path / "ckpt"), run_id="r")
        assert saved_ok(out)
        assert "BACKEND-F-007: failed to register checkpoint in manifest" in caplog.text


# ---------------------------------------------------------------------------
# export / release / push
# ---------------------------------------------------------------------------

class TestExportRouting:
    def _t(self, tmp_path):
        t = make(tmp_path, mode="full", learning_rate=1e-5)
        t._model, t._tokenizer, t._is_loaded = tiny_llama(layers=1), tiny_tokenizer(), True
        return t

    def test_non_peft_merged_export_uses_in_memory_model(self, tmp_path):
        t = self._t(tmp_path)
        result = SimpleNamespace(path=str(tmp_path / "merged"))
        with patch("backpropagate.export.export_merged", return_value=result) as m:
            out = t.export("merged", push_to_hub=True, repo_id="org/m", safe=True)
        assert out is result
        kw = m.call_args.kwargs
        assert kw["model"] is t._model and kw["tokenizer"] is t._tokenizer
        assert kw["push_to_hub"] is True and kw["repo_id"] == "org/m" and kw["safe"] is True
        assert Path(kw["output_dir"]) == Path(t.output_dir) / "merged"

    def test_non_peft_gguf_export_forwards_quantization(self, tmp_path):
        t = self._t(tmp_path)
        result = SimpleNamespace(path=str(tmp_path / "g"))
        with patch("backpropagate.export.export_gguf", return_value=result) as m:
            out = t.export("GGUF", output_dir=str(tmp_path / "g"), quantization="q8_0")
        assert out is result and m.call_args.kwargs["quantization"] == "q8_0"
        assert Path(m.call_args.kwargs["output_dir"]) == tmp_path / "g"

    def test_unknown_format_and_unloaded_model_are_rejected(self, tmp_path):
        t = self._t(tmp_path)
        with pytest.raises(ValueError, match="Unsupported format: onnx"):
            t.export("onnx")
        t._is_loaded = False
        with pytest.raises(TrainingError) as ei:
            t.export("lora")
        assert ei.value.code == "RUNTIME_EXPORT_FAILED"


class TestReleaseModel:
    def test_release_drops_state_and_empties_cuda_cache_when_available(self, tmp_path):
        t = make(tmp_path)
        t._model, t._trainer, t._is_loaded = object(), object(), True
        calls = []
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.empty_cache", side_effect=lambda: calls.append(1)):
            t._release_model()
        assert (t._model, t._trainer, t._is_loaded) == (None, None, False)
        assert calls == [1]

    def test_cache_reclaim_failure_is_swallowed(self, tmp_path):
        t = make(tmp_path)
        t._model, t._is_loaded = object(), True
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.empty_cache", side_effect=RuntimeError("context lost")):
            t._release_model()
        assert t._model is None

    def test_push_to_hub_requires_a_loaded_model_then_pushes_both(self, tmp_path):
        t = make(tmp_path)
        with pytest.raises(TrainingError, match="Cannot push to Hub"):
            t.push_to_hub("org/x")
        pushed = []
        t._model = SimpleNamespace(push_to_hub=lambda repo, private: pushed.append(("model", repo, private)))
        t._tokenizer = SimpleNamespace(push_to_hub=lambda repo, private: pushed.append(("tok", repo, private)))
        t._is_loaded = True
        t.push_to_hub("org/x", private=False)
        assert pushed == [("model", "org/x", False), ("tok", "org/x", False)]
        assert t.model is t._model and t.tokenizer is t._tokenizer and t.runs == []


# ---------------------------------------------------------------------------
# multi_run delegation
# ---------------------------------------------------------------------------

class TestMultiRunDelegation:
    def test_unsafe_gpu_refuses_before_building_anything(self, tmp_path):
        t = make(tmp_path)
        with patch.object(T, "check_gpu_safe", return_value=False):
            with pytest.raises(GPUNotAvailableError) as ei:
                t.multi_run("data.jsonl")
        assert ei.value.code == "DEP_GPU_NOT_AVAILABLE"
        assert "GPU safety check failed" in ei.value.suggestion

    def test_mlx_backend_refuses_multi_run(self, tmp_path):
        t = make(tmp_path)
        t._effective_backend = "mlx"
        with pytest.raises(InvalidSettingError) as ei:
            t.multi_run("data.jsonl")
        assert ei.value.setting_name == "backend"

    def test_config_is_assembled_from_the_trainer_and_run_is_delegated(self, tmp_path):
        built = {}

        class FakeMulti:
            def __init__(self, **kw):
                built.update(kw)

            def run(self, dataset):
                built["dataset"] = dataset
                return "RESULT"

        t = make(tmp_path, mode="full", learning_rate=2e-5)
        hook = object()
        with patch.object(T, "check_gpu_safe", return_value=True), \
                patch("backpropagate.multi_run.MultiRunTrainer", FakeMulti):
            out = t.multi_run("data.jsonl", num_runs=3, steps_per_run=7, samples_per_run=11,
                              merge_mode="SIMPLE", on_run_complete=hook, resume_from="abc")
        assert out == "RESULT" and built["dataset"] == "data.jsonl"
        cfg = built["config"]
        assert (cfg.num_runs, cfg.steps_per_run, cfg.samples_per_run) == (3, 7, 11)
        assert cfg.merge_mode.value == "simple" and cfg.mode == "full"
        assert cfg.initial_lr == pytest.approx(2e-5)
        assert Path(cfg.checkpoint_dir) == Path(t.output_dir) / "multi_run"
        assert built["on_run_complete"] is hook and built["resume_from"] == "abc"
        assert built["model"] == "acme/Tiny-1B"

    def test_explicit_checkpoint_dir_wins(self, tmp_path):
        built = {}

        class FakeMulti:
            def __init__(self, **kw):
                built.update(kw)

            def run(self, dataset):
                return None

        t = make(tmp_path)
        with patch.object(T, "check_gpu_safe", return_value=True), \
                patch("backpropagate.multi_run.MultiRunTrainer", FakeMulti):
            t.multi_run("d", checkpoint_dir=str(tmp_path / "elsewhere"))
        assert built["config"].checkpoint_dir == str(tmp_path / "elsewhere")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

class TestModuleFunctions:
    def test_load_model_wrapper_returns_model_and_tokenizer(self):
        def fake_load(self):
            self._model, self._tokenizer = "M", "TOK"

        with patch.object(Trainer, "load_model", fake_load):
            assert T.load_model("acme/Tiny-1B", max_seq_length=64) == ("M", "TOK")

    def test_load_dataset_reads_json_jsonl_and_csv_files(self, tmp_path):
        jl = write_jsonl(tmp_path / "a.jsonl", [{"x": 1}, {"x": 2}])
        assert list(T.load_dataset(jl)["x"]) == [1, 2]
        js = tmp_path / "b.json"
        js.write_text('{"x": 3}\n{"x": 4}\n', encoding="utf-8")
        assert list(T.load_dataset(str(js))["x"]) == [3, 4]
        csv = tmp_path / "c.csv"
        csv.write_text("x,y\n5,a\n6,b\n", encoding="utf-8")
        assert list(T.load_dataset(str(csv))["x"]) == [5, 6]

    def test_load_dataset_hub_name_uses_requested_split(self):
        ds = Dataset.from_dict({"x": [1]})
        with patch("datasets.load_dataset", return_value=ds) as m:
            assert T.load_dataset("org/name", split="validation") is ds
            T.load_dataset("org/name")
        assert m.call_args_list[0].args == ("org/name",) and m.call_args_list[0].kwargs == {"split": "validation"}
        assert m.call_args_list[1].kwargs == {"split": "train"}

    def test_load_dataset_passes_through_in_memory_datasets_and_caps_samples(self):
        ds = Dataset.from_dict({"x": list(range(10))})
        assert T.load_dataset(ds) is ds
        capped = T.load_dataset(ds, max_samples=3)
        assert len(capped) == 3 and set(capped["x"]) <= set(range(10))
        assert len(T.load_dataset(ds, max_samples=50)) == 10

    def test_lazy_multi_run_exports(self):
        from backpropagate import multi_run

        assert T.MultiRunTrainer is multi_run.MultiRunTrainer
        assert T.SpeedrunTrainer is multi_run.SpeedrunTrainer
        with pytest.raises(AttributeError, match="no attribute 'Nope'"):
            T.Nope  # noqa: B018

    def test_trainingrun_defaults(self):
        run = T.TrainingRun(run_id="r", steps=1, final_loss=0.5)
        assert run.loss_history == [] and run.metadata == {} and run.output_path is None
        assert math.isfinite(run.final_loss)
