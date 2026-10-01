"""Remaining coverage for ``backpropagate.trainer``: the MLX run wrapper, the
Unsloth response-masking helper on POSIX, per-card optimizer double-probe, KTO
config knobs, sample caps and OOM detection without a CUDA exception type.

Mock boundary (named per test): Apple-Silicon detection (``platform``), the
``mlx_lm.lora`` subprocess (``MLXBackend.run``) and its data-dir writer,
Unsloth's ``chat_templates`` module (CUDA-only optional library; a fake module
in ``sys.modules``) and trl's ``KTOConfig`` (version-dependent dataclass).
Real tiny CPU training is used for the sample-cap and OOM-detection tests.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from backpropagate import trainer as T  # noqa: E402
from backpropagate.checkpoints import RunHistoryManager  # noqa: E402
from backpropagate.exceptions import DatasetError, InvalidSettingError  # noqa: E402
from backpropagate.mlx_backend import MLXBackend, MLXRunResult  # noqa: E402
from backpropagate.trainer import Trainer, TrainingCallback  # noqa: E402
from tests.helpers.tiny_models import sentences, tiny_llama, tiny_tokenizer  # noqa: E402

LOGGER = "backpropagate.trainer"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(T, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(T, "_RETRY_MAX_SECONDS", 0)


# ---------------------------------------------------------------------------
# MLX rail
# ---------------------------------------------------------------------------

@pytest.fixture
def apple():
    """Pretend to be an Apple-Silicon Mac that has the mlx extra."""
    patches = [
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch("platform.machine", return_value="arm64"),
        mock.patch("backpropagate.mlx_backend.check_feature", return_value=True),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def mlx_trainer(tmp_path, **kw):
    return Trainer(backend="mlx", model="mlx-community/tiny", output_dir=str(tmp_path / "out"),
                   use_unsloth=False, report_to="none", **kw)


def mlx_result(tmp_path, loss=0.75, val=0.8):
    return MLXRunResult(adapter_path=str(tmp_path / "out" / "mlx_adapter"), final_loss=loss,
                        iters=6, raw_stdout="Iter 6: Train loss 0.750", val_loss=val)


class TestMlxRail:
    def test_train_routes_to_the_mlx_rail_and_records_history(self, apple, tmp_path):
        t = mlx_trainer(tmp_path)
        done = []
        cb = TrainingCallback(on_complete=done.append)
        with patch("backpropagate.mlx_backend.prepare_mlx_data_dir") as prep, \
                patch.object(MLXBackend, "run", return_value=mlx_result(tmp_path)):
            run = t.train("data.jsonl", steps=6, samples=20, callback=cb)
        assert run.final_loss == pytest.approx(0.75) and run.steps == 6
        assert run.metadata["backend"] == "mlx" and run.metadata["val_loss"] == 0.8
        assert run.metadata["final_loss_parsed"] is True
        assert t._has_trained is True and t.runs == [run] and done == [run]
        assert prep.call_args.args[0] == "data.jsonl" and prep.call_args.kwargs["max_samples"] == 20
        (row,) = RunHistoryManager(str(tmp_path / "out")).get_history()
        assert row["status"] == "completed" and row["hyperparameters"]["backend"] == "mlx"
        assert row["final_loss"] == pytest.approx(0.75)
        assert t._is_loaded is False  # the MLX rail never loads a model in-process

    def test_mlx_callback_and_history_failures_never_break_a_successful_run(self, apple, tmp_path, caplog):
        t = mlx_trainer(tmp_path)

        def bad(run):
            raise RuntimeError("dashboard down")

        with patch("backpropagate.mlx_backend.prepare_mlx_data_dir"), \
                patch.object(MLXBackend, "run", return_value=mlx_result(tmp_path)), \
                patch.object(RunHistoryManager, "record_run_started", side_effect=OSError("h1")), \
                patch.object(RunHistoryManager, "record_run_completed", side_effect=OSError("h2")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t._train_with_mlx("d.jsonl", steps=6, samples=None,
                                    callback=TrainingCallback(on_complete=bad))
        assert run.steps == 6
        assert "record_run_started failed: h1" in caplog.text
        assert "record_run_completed failed: h2" in caplog.text
        assert "on_complete callback raised error: dashboard down" in caplog.text

    def test_mlx_failure_is_recorded_reported_and_reraised(self, apple, tmp_path, caplog):
        t = mlx_trainer(tmp_path)
        errors = []

        def bad_on_error(exc):
            errors.append(exc)
            raise RuntimeError("alerting down")

        boom = DatasetError("unreadable dataset")
        with patch("backpropagate.mlx_backend.prepare_mlx_data_dir", side_effect=boom), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(DatasetError) as ei:
                t._train_with_mlx("d.jsonl", steps=3, samples=None,
                                  callback=TrainingCallback(on_error=bad_on_error))
        assert ei.value is boom and errors == [boom]
        assert "on_error callback raised error: alerting down" in caplog.text
        (row,) = RunHistoryManager(str(tmp_path / "out")).get_history()
        assert row["status"] == "failed" and "unreadable dataset" in row["failure_reason"]

    def test_mlx_failure_with_broken_history_and_no_callback_still_reraises(self, apple, tmp_path, caplog):
        t = mlx_trainer(tmp_path)
        with patch("backpropagate.mlx_backend.prepare_mlx_data_dir", return_value=None), \
                patch.object(MLXBackend, "run", side_effect=RuntimeError("mlx_lm exited 1")), \
                patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(RuntimeError, match="mlx_lm exited 1"):
                t._train_with_mlx("d.jsonl", steps=3, samples=None, callback=None)
        assert "record_run_failed failed: ro" in caplog.text

    def test_multi_run_is_refused_on_the_mlx_rail(self, apple, tmp_path):
        with pytest.raises(InvalidSettingError):
            mlx_trainer(tmp_path).multi_run("d.jsonl")


# ---------------------------------------------------------------------------
# Unsloth response masking (POSIX path)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_unsloth_chat(monkeypatch):
    calls = {}

    def masker(trainer, instruction_part, response_part, num_proc):
        calls.update(instruction_part=instruction_part, response_part=response_part, num_proc=num_proc)
        return SimpleNamespace(wrapped_of=trainer)

    pkg = types.ModuleType("unsloth")
    chat = types.ModuleType("unsloth.chat_templates")
    chat.train_on_responses_only = masker
    pkg.chat_templates = chat
    monkeypatch.setitem(sys.modules, "unsloth", pkg)
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", chat)
    return calls


class TestApplyTrainOnResponsesOnly:
    def test_disabled_or_non_unsloth_returns_trainer_untouched(self):
        sentinel = object()
        assert T._apply_train_on_responses_only(sentinel, None, enabled=False, use_unsloth=True) == (sentinel, None)
        assert T._apply_train_on_responses_only(sentinel, None, enabled=True, use_unsloth=False) == (sentinel, None)

    def test_windows_skips_with_warning(self, caplog):
        sentinel = object()
        with patch("backpropagate.trainer.os.name", "nt"), caplog.at_level(logging.WARNING, logger=LOGGER):
            out = T._apply_train_on_responses_only(sentinel, None, enabled=True, use_unsloth=True)
        assert out == (sentinel, None) and "disabled on Windows" in caplog.text

    def test_operator_override_wins_over_detection(self, fake_unsloth_chat, caplog):
        with patch("backpropagate.trainer.os.name", "posix"), caplog.at_level(logging.INFO, logger=LOGGER):
            wrapped, markers = T._apply_train_on_responses_only(
                "TRAINER", object(), enabled=True, use_unsloth=True,
                response_markers_override=("<U>", "<A>"))
        assert markers == ("<U>", "<A>") and wrapped.wrapped_of == "TRAINER"
        assert fake_unsloth_chat == {"instruction_part": "<U>", "response_part": "<A>", "num_proc": 1}
        assert "using operator override" in caplog.text

    def test_markers_are_detected_from_the_tokenizer_family(self, fake_unsloth_chat):
        tok = SimpleNamespace(name_or_path="Qwen/Qwen2.5-1.5B")
        with patch("backpropagate.trainer.os.name", "posix"):
            wrapped, markers = T._apply_train_on_responses_only(
                "T", tok, enabled=True, use_unsloth=True)
        assert markers == ("<|im_start|>user", "<|im_start|>assistant")
        assert fake_unsloth_chat["instruction_part"] == "<|im_start|>user"

    def test_old_unsloth_without_the_masker_degrades_with_warning(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "unsloth.chat_templates", None)
        monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
        with patch("backpropagate.trainer.os.name", "posix"), caplog.at_level(logging.WARNING, logger=LOGGER):
            out = T._apply_train_on_responses_only("T", object(), enabled=True, use_unsloth=True)
        assert out == ("T", None)
        assert "not available in this Unsloth version" in caplog.text

    def test_masker_crash_degrades_with_warning(self, monkeypatch, caplog):
        chat = types.ModuleType("unsloth.chat_templates")

        def explode(*a, **k):
            raise ValueError("template has no assistant turn")

        chat.train_on_responses_only = explode
        monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))
        monkeypatch.setitem(sys.modules, "unsloth.chat_templates", chat)
        with patch("backpropagate.trainer.os.name", "posix"), caplog.at_level(logging.WARNING, logger=LOGGER):
            out = T._apply_train_on_responses_only(
                "T", object(), enabled=True, use_unsloth=True, response_markers_override=("a", "b"))
        assert out == ("T", None)
        assert "Failed to apply train_on_responses_only: template has no assistant turn" in caplog.text


# ---------------------------------------------------------------------------
# optimizer double-probe + KTO config knobs
# ---------------------------------------------------------------------------

class TestOptimDoubleProbe:
    def test_cuda_vanishing_between_the_two_probes_still_downgrades(self):
        # Rule 1 sees CUDA, rule 3's own probe does not: still adamw_torch.
        with patch("torch.cuda.is_available", side_effect=[True, False]):
            assert Trainer._detect_optim_for_card("adamw_8bit") == "adamw_torch"


class TestKtoConfigKnobs:
    def _fake_kto_config(self):
        names = ["output_dir", "per_device_train_batch_size", "gradient_accumulation_steps", "max_steps",
                 "learning_rate", "warmup_steps", "optim", "lr_scheduler_type", "logging_steps", "bf16",
                 "fp16", "seed", "dataloader_num_workers", "report_to", "run_name", "max_length", "beta",
                 "desirable_weight", "undesirable_weight", "train_sampling_strategy", "save_steps",
                 "weight_decay", "max_prompt_length", "max_completion_length"]
        return dataclasses.make_dataclass("KTOConfig", [(n, object, dataclasses.field(default=None))
                                                         for n in names])

    def _call(self, **kw):
        return T._build_kto_config(
            output_dir="o", per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=5,
            learning_rate=1e-6, warmup_steps=0, max_seq_length=1024, kto_beta=0.1,
            desirable_weight=1.0, undesirable_weight=1.2, seed=1, lr_scheduler_type="linear",
            logging_steps=1, **kw)

    def test_explicit_lengths_are_forwarded_when_the_config_declares_them(self):
        with patch("trl.KTOConfig", self._fake_kto_config()):
            cfg = self._call(max_prompt_length=300, max_completion_length=200)
        assert cfg.max_prompt_length == 300 and cfg.max_completion_length == 200
        assert cfg.beta == 0.1 and cfg.undesirable_weight == 1.2
        assert cfg.train_sampling_strategy == "sequential" and cfg.max_length == 1024

    def test_short_window_derives_half_when_declared(self):
        with patch("trl.KTOConfig", self._fake_kto_config()):
            cfg = T._build_kto_config(
                output_dir="o", per_device_train_batch_size=1, gradient_accumulation_steps=1,
                max_steps=1, learning_rate=1e-6, warmup_steps=0, max_seq_length=64, kto_beta=0.1,
                desirable_weight=1.0, undesirable_weight=1.0, seed=1, lr_scheduler_type="linear",
                logging_steps=1)
        assert cfg.max_prompt_length == 32 and cfg.max_completion_length is None


# ---------------------------------------------------------------------------
# Real tiny CPU training: valid sample cap, OOM detection without a CUDA type
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("edge_tiny")
    tiny_llama(layers=2).save_pretrained(d)
    tiny_tokenizer().save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def chat_jsonl(tmp_path_factory):
    p = tmp_path_factory.mktemp("edge_data") / "chat.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for s in sentences(16):
            fh.write(json.dumps({"messages": [{"role": "user", "content": s},
                                              {"role": "assistant", "content": s}]}) + "\n")
    return str(p)


def real_trainer(tiny_dir, tmp_path, **kw):
    base = {"model": tiny_dir, "use_unsloth": False, "load_in_4bit": False, "batch_size": 2,
            "max_seq_length": 32, "output_dir": str(tmp_path / "out"), "learning_rate": 1e-3,
            "packing": False, "report_to": "none", "lora_r": 4}
    base.update(kw)
    return Trainer(**base)


class TestRealRunEdges:
    def test_valid_sample_cap_limits_the_training_set(self, tiny_dir, chat_jsonl, tmp_path):
        run = real_trainer(tiny_dir, tmp_path).train(chat_jsonl, steps=1, samples=6)
        assert run.samples_seen == 6

    def test_oom_is_detected_by_message_when_torch_lacks_the_cuda_oom_type(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        from trl import SFTTrainer

        monkeypatch.delattr(torch.cuda, "OutOfMemoryError")
        real_train = SFTTrainer.train
        state = {"n": 0}

        def flaky(inner, *a, **k):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
            return real_train(inner, *a, **k)

        monkeypatch.setattr(SFTTrainer, "train", flaky)
        t = real_trainer(tiny_dir, tmp_path, batch_size=2)
        run = t.train(chat_jsonl, steps=1)
        assert run.metadata["oom_retries"] == 1
        assert os.path.isdir(tmp_path / "out")


# ---------------------------------------------------------------------------
# Leftover data / save branches
# ---------------------------------------------------------------------------

class TestLeftoverBranches:
    def _t(self, tmp_path, **kw):
        base = {"model": "acme/Tiny-1B", "use_unsloth": False, "batch_size": 2, "max_seq_length": 32,
                "output_dir": str(tmp_path / "out"), "report_to": "none"}
        base.update(kw)
        return Trainer(**base)

    def test_hub_name_sft_and_valid_kto_datasets_pass_through(self, tmp_path):
        from datasets import Dataset

        text = Dataset.from_dict({"text": ["a", "b"]})
        kto = Dataset.from_list([{"prompt": "q", "completion": "a", "label": True}])
        with patch("datasets.load_dataset", return_value=text):
            assert len(self._t(tmp_path)._load_dataset("org/sft", samples=0)) == 2
        with patch("datasets.load_dataset", return_value=kto):
            t = self._t(tmp_path, method="kto", learning_rate=1e-6)
            assert t._load_dataset("org/kto", samples=0, method="kto").column_names == [
                "prompt", "completion", "label"]
        # in-memory KTO dataset with the right columns
        assert len(t._load_dataset(kto, samples=0, method="kto")) == 1

    def test_trace_probe_returning_a_non_string_is_ignored(self, tmp_path):
        from datasets import Dataset

        t = self._t(tmp_path, reasoning_trace=True)

        class ListTemplate:
            def encode(self, text):
                return text.split()

            def apply_chat_template(self, *a, **k):
                return ["not", "a", "string"]

        t._tokenizer = ListTemplate()
        ds = Dataset.from_dict({"text": ["<think>" + " ".join(["w"] * 12) + "</think> done", "no think"]})
        assert len(t._filter_reasoning_traces(ds)) == 1

    def test_first_ever_save_failing_to_promote_leaves_no_partial(self, tmp_path):
        import shutil

        t = self._t(tmp_path, mode="full", learning_rate=1e-5)
        t._model, t._tokenizer = tiny_llama(layers=1), tiny_tokenizer()
        t._is_loaded = t._has_trained = True
        real_move = shutil.move

        def fail(src, dst, *a, **k):
            if str(src).endswith(".partial"):
                raise OSError("target volume offline")
            return real_move(src, dst, *a, **k)

        from backpropagate.exceptions import CheckpointError

        with patch("shutil.move", side_effect=fail):
            with pytest.raises(CheckpointError, match="target volume offline"):
                t.save(str(tmp_path / "ckpt"), register_in_manifest=False)
        assert not (tmp_path / "ckpt").exists() and not (tmp_path / "ckpt.partial").exists()
