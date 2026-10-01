"""Coverage tests for ``Trainer.train``: the real CPU training loop, run-history
bookkeeping, resume, OOM recovery and failure routing.

Training is REAL: a tiny random-weight Llama (``tests/helpers/tiny_models.py``)
is trained with LoRA through the actual ``trl.SFTTrainer`` on CPU for 2-3 steps
(bf16 base, no bitsandbytes), then history / callbacks / metadata are checked.

Mock boundary: GPU failure injection. A CUDA OOM cannot occur on a CPU runner,
so ``SFTTrainer.train`` is wrapped to raise a real ``torch.cuda.OutOfMemoryError``
(or CUDA-shaped ``RuntimeError``) on the first attempts and then run the real
training; the CUDA availability / ``empty_cache`` / gpu_safety probes that the
recovery path consults are stubbed. Unsloth's masking utility is the other
boundary (``_apply_train_on_responses_only``).
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate import trainer as T  # noqa: E402
from backpropagate.checkpoints import RunHistoryManager  # noqa: E402
from backpropagate.exceptions import (  # noqa: E402
    BackpropagateError,
    GPUMemoryError,
    InvalidSettingError,
    TrainingAbortedError,
    TrainingError,
)
from backpropagate.trainer import Trainer, TrainingCallback  # noqa: E402
from tests.helpers.tiny_models import sentences, tiny_llama, tiny_tokenizer  # noqa: E402

LOGGER = "backpropagate.trainer"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("train_tiny")
    tiny_llama(layers=2).save_pretrained(d)
    tiny_tokenizer().save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def chat_jsonl(tmp_path_factory):
    p = tmp_path_factory.mktemp("train_data") / "chat.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for s in sentences(16):
            fh.write(json.dumps({"messages": [{"role": "user", "content": s},
                                              {"role": "assistant", "content": s}]}) + "\n")
    return str(p)


@pytest.fixture(autouse=True)
def _fast_settings(monkeypatch):
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
    monkeypatch.setattr(T.settings.training, "logging_steps", 1)
    monkeypatch.setattr(T.settings.training, "save_steps", 2)
    monkeypatch.setattr(T, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(T, "_RETRY_MAX_SECONDS", 0)


def make(tiny_dir, tmp_path, **kw):
    base = {
        "model": tiny_dir, "use_unsloth": False, "load_in_4bit": False, "batch_size": 2,
        "max_seq_length": 32, "output_dir": str(tmp_path / "out"), "learning_rate": 1e-3,
        "packing": False, "report_to": "none", "lora_r": 4,
    }
    base.update(kw)
    return Trainer(**base)


def history(tmp_path):
    return RunHistoryManager(str(tmp_path / "out")).get_history()


class FlakyTrain:
    """Wrap ``SFTTrainer.train``: raise the queued exceptions, then train for real.

    ``after`` (optional) post-processes the real result: ``after(result, inner)``.
    ``before_raise`` runs just before each injected exception is raised.
    """

    def __init__(self, monkeypatch, exceptions=(), after=None, before_raise=None):
        from trl import SFTTrainer

        self.real = SFTTrainer.train
        self.pending = list(exceptions)
        self.calls = 0
        self.kwargs = []

        def flaky(inner, *a, **k):
            self.calls += 1
            self.kwargs.append(k)
            if self.pending:
                exc = self.pending.pop(0)
                if before_raise is not None:
                    before_raise()
                raise exc
            result = self.real(inner, *a, **k)
            return after(result, inner) if after is not None else result

        monkeypatch.setattr(SFTTrainer, "train", flaky)


def oom():
    return torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestTrainEndToEnd:
    def test_lora_run_records_history_callbacks_and_finite_loss_curve(
        self, tiny_dir, chat_jsonl, tmp_path
    ):
        steps, epochs, saves, completes, errors = [], [], [], [], []
        cb = TrainingCallback(
            on_step=lambda s, loss: steps.append((s, loss)),
            on_epoch=epochs.append,
            on_save=saves.append,
            on_complete=completes.append,
            on_error=errors.append,
        )
        t = make(tiny_dir, tmp_path)
        run = t.train(chat_jsonl, steps=3, callback=cb)

        assert run.steps == 3 and run.samples_seen == 16
        assert len(run.loss_history) == 3 and all(0 < x < 100 for x in run.loss_history)
        assert run.final_loss > 0
        assert run.metadata["oom_retries"] == 0
        assert run.metadata["effective_batch_size"] == 2
        assert t.runs == [run] and t._has_trained is True
        # one on_step per logged loss; the closing summary log re-reports the last step
        assert [s for s, _ in steps][:3] == [1, 2, 3]
        assert all(loss > 0 for _, loss in steps)
        assert completes == [run] and errors == []
        assert epochs  # at least one epoch boundary fired
        assert any(p.endswith("checkpoint-2") for p in saves)

        (row,) = history(tmp_path)
        assert row["run_id"] == run.run_id and row["status"] == "completed"
        assert row["final_loss"] == pytest.approx(run.final_loss)
        assert row["steps"] == 3 and row["session_kind"] == "single_run"
        assert row["hyperparameters"]["lora_r"] == 4
        assert row["hyperparameters"]["fp8"] is False
        assert "response_markers" not in row["hyperparameters"]  # masking never ran
        assert row["dataset_hash"] and len(row["dataset_hash"]) == 16

    def test_response_markers_land_in_run_history(self, tiny_dir, chat_jsonl, tmp_path):
        t = make(tiny_dir, tmp_path)

        def fake_mask(trainer, tokenizer, **kw):
            return trainer, ("<|user|>", "<|assistant|>")

        with patch.object(T, "_apply_train_on_responses_only", side_effect=fake_mask) as m:
            t.train(chat_jsonl, steps=1)
        assert m.call_args.kwargs["enabled"] is True and m.call_args.kwargs["use_unsloth"] is False
        (row,) = history(tmp_path)
        assert row["hyperparameters"]["response_markers"] == ["<|user|>", "<|assistant|>"]

    def test_history_start_and_completion_failures_never_kill_training(
        self, tiny_dir, chat_jsonl, tmp_path, caplog
    ):
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "record_run_started", side_effect=OSError("disk gone")), \
                patch.object(RunHistoryManager, "record_run_completed", side_effect=OSError("still gone")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert run.steps == 1
        assert "RunHistoryManager.record_run_started failed: disk gone" in caplog.text
        assert "RunHistoryManager.record_run_completed failed: still gone" in caplog.text

    def test_on_complete_callback_error_is_isolated(self, tiny_dir, chat_jsonl, tmp_path, caplog):
        def bad(run):
            raise RuntimeError("dashboard down")

        t = make(tiny_dir, tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1, callback=TrainingCallback(on_complete=bad))
        assert run.steps == 1
        assert "on_complete callback raised error: dashboard down" in caplog.text

    @pytest.mark.parametrize("kw", [{"steps": 0}, {"steps": -1}, {"steps": "5"},
                                    {"samples": 0}, {"samples": -4}, {"samples": 1.5}])
    def test_invalid_steps_or_samples_rejected_before_any_work(self, tiny_dir, tmp_path, kw):
        t = make(tiny_dir, tmp_path)
        with pytest.raises(InvalidSettingError) as ei:
            t.train("x.jsonl", **kw)
        assert ei.value.setting_name in ("steps", "samples")
        assert t._is_loaded is False


# ---------------------------------------------------------------------------
# Result-shape tolerance
# ---------------------------------------------------------------------------

class TestResultCoercion:
    def test_missing_training_loss_attribute_records_zero_with_warning(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        FlakyTrain(monkeypatch, after=lambda res, inner: SimpleNamespace())
        t = make(tiny_dir, tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert run.final_loss == 0.0
        assert "missing 'training_loss' attribute" in caplog.text

    def test_non_numeric_and_non_finite_losses_are_coerced(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        def after(res, inner):
            inner.state.log_history.extend([
                {"loss": "garbage"}, {"loss": float("nan")}, {"loss": 1.25}, {"eval_loss": 9.0},
            ])
            return SimpleNamespace(training_loss="not-a-number")

        FlakyTrain(monkeypatch, after=after)
        t = make(tiny_dir, tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert run.final_loss == 0.0
        assert "Non-numeric final training_loss ('not-a-number')" in caplog.text
        assert 1.25 in run.loss_history
        assert all(x == x and abs(x) != float("inf") for x in run.loss_history)  # no NaN/inf
        assert "garbage" not in repr(run.loss_history)

    def test_nan_final_loss_is_replaced_with_zero(self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog):
        FlakyTrain(monkeypatch, after=lambda res, inner: SimpleNamespace(training_loss=float("inf")))
        t = make(tiny_dir, tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert run.final_loss == 0.0
        assert "Non-finite final training_loss" in caplog.text


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    def _record(self, tmp_path, run_id="prior", **fields):
        h = RunHistoryManager(str(tmp_path / "out"))
        h.record_run_started(run_id, model_name="m", **fields)
        return h

    def test_unknown_run_id_is_a_hard_error(self, tiny_dir, chat_jsonl, tmp_path):
        t = make(tiny_dir, tmp_path)
        with pytest.raises(InvalidSettingError) as ei:
            t.train(chat_jsonl, steps=1, resume_from="nope")
        assert ei.value.setting_name == "resume_from"
        assert "backprop runs" in ei.value.suggestion

    def test_missing_checkpoint_marks_prior_failed_and_starts_fresh(
        self, tiny_dir, chat_jsonl, tmp_path, caplog
    ):
        self._record(tmp_path, checkpoint_path=str(tmp_path / "gone"))
        t = make(tiny_dir, tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1, resume_from="prior")
        assert run.run_id != "prior"
        assert "no longer exists on disk" in caplog.text
        rows = {r["run_id"]: r for r in history(tmp_path)}
        assert rows["prior"]["status"] == "failed"
        assert rows["prior"]["failure_reason"] == "resume_checkpoint_missing"
        assert rows[run.run_id]["status"] == "completed"

    def test_failure_to_mark_stale_entry_is_only_a_warning(
        self, tiny_dir, chat_jsonl, tmp_path, caplog
    ):
        self._record(tmp_path, checkpoint_path=str(tmp_path / "gone"))
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro fs")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1, resume_from="prior")
        assert run.run_id != "prior"
        assert "Failed to mark stale resume entry as failed: ro fs" in caplog.text

    def test_infrastructure_failure_in_lookup_degrades_to_fresh_run(
        self, tiny_dir, chat_jsonl, tmp_path, caplog
    ):
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "get_run", side_effect=ValueError("corrupt json")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1, resume_from="anything")
        assert run.steps == 1
        assert "Resume lookup failed: corrupt json" in caplog.text

    def test_record_without_checkpoint_reuses_run_id_and_flips_status_back(
        self, tiny_dir, chat_jsonl, tmp_path
    ):
        h = self._record(tmp_path, checkpoint_path=None)
        h.record_run_failed("prior", "earlier crash")
        t = make(tiny_dir, tmp_path)
        run = t.train(chat_jsonl, steps=1, resume_from="prior")
        assert run.run_id == "prior"
        rows = history(tmp_path)
        assert [r["run_id"] for r in rows] == ["prior"]
        assert rows[0]["status"] == "completed"  # running -> completed in place

    def test_checkpoint_dir_without_any_checkpoint_starts_fresh_with_warning(
        self, tiny_dir, chat_jsonl, tmp_path, caplog
    ):
        empty = tmp_path / "empty_ckpt_dir"
        empty.mkdir()
        self._record(tmp_path, checkpoint_path=str(empty))
        t = make(tiny_dir, tmp_path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1, resume_from="prior")
        assert run.run_id == "prior"
        assert "holds no checkpoint-<step> directory" in caplog.text

    def test_resume_checkpoint_path_vanishing_before_train_falls_back_to_fresh(
        self, tiny_dir, chat_jsonl, tmp_path, caplog
    ):
        """The lookup saw the path, but it is gone by the time the loop checks it."""
        keep = tmp_path / "ckpt_dir"
        keep.mkdir()
        self._record(tmp_path, checkpoint_path=str(keep))
        t = make(tiny_dir, tmp_path)
        real_exists = T.Path.exists
        state = {"n": 0}

        def vanishing(self):
            if str(self) == str(keep):
                state["n"] += 1
                if state["n"] > 1:  # first call: the stale-record guard; later: the loop
                    return False
            return real_exists(self)

        with patch.object(T.Path, "exists", vanishing), caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1, resume_from="prior")
        assert run.run_id == "prior"
        assert "no longer exists on disk; starting fresh" in caplog.text


# ---------------------------------------------------------------------------
# OOM recovery (real training after injected CUDA failures)
# ---------------------------------------------------------------------------

class TestOomRecovery:
    def test_cuda_oom_halves_batch_trains_for_real_and_patches_history(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        flaky = FlakyTrain(monkeypatch, exceptions=[oom()])
        t = make(tiny_dir, tmp_path, batch_size=4, gradient_accumulation=1)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=2)
        assert flaky.calls == 2
        assert run.metadata["oom_retries"] == 1
        assert run.metadata["effective_batch_size"] == 2
        assert run.metadata["effective_gradient_accumulation"] == 2
        assert (t.batch_size, t.gradient_accumulation) == (4, 1)  # configured values restored
        assert "event=oom_recovery_adjust" in caplog.text and "(preserved)" in caplog.text
        (row,) = history(tmp_path)
        assert row["effective_batch_size"] == 2 and row["oom_retries"] == 1
        assert row["status"] == "completed"

    def test_effective_batch_history_patch_failure_is_a_warning(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        FlakyTrain(monkeypatch, exceptions=[oom()])
        t = make(tiny_dir, tmp_path, batch_size=4)
        with patch.object(RunHistoryManager, "update_run", side_effect=OSError("locked")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert run.metadata["oom_retries"] == 1
        assert "RunHistoryManager.update_run (effective batch) failed: locked" in caplog.text

    @pytest.mark.parametrize("marker_text,marker", [
        ("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate", "cublas_status_alloc_failed"),
        ("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED", "cudnn_status_not_initialized"),
    ])
    def test_oom_adjacent_library_errors_are_recovered_with_a_structured_warning(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog, marker_text, marker
    ):
        FlakyTrain(monkeypatch, exceptions=[RuntimeError(marker_text)])
        t = make(tiny_dir, tmp_path, batch_size=2)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert run.metadata["oom_retries"] == 1
        assert f"marker={marker!r}" in caplog.text
        assert "code=RUNTIME_OOM_ADJACENT" in caplog.text

    def test_recovery_disabled_wraps_oom_with_run_id_and_vram_snapshot(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        from backpropagate.gpu_safety import GPUStatus

        FlakyTrain(monkeypatch, exceptions=[RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")])
        snap = GPUStatus(available=True, vram_total_gb=16.0, vram_used_gb=14.0, vram_free_gb=2.0)
        t = make(tiny_dir, tmp_path, oom_recovery=False)
        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=snap):
            with pytest.raises(GPUMemoryError) as ei:
                t.train(chat_jsonl, steps=1)
        d = ei.value.details
        assert ei.value.code == "RUNTIME_GPU_OOM"
        assert d["adjacent_marker"] == "cublas_status_alloc_failed"
        assert d["vram_total_gb"] == 16.0 and d["vram_scope"] == "per_process"
        assert d["vram_free_gb_this_process"] == 2.0
        assert d["run_id"] == history(tmp_path)[0]["run_id"]
        assert isinstance(ei.value.__cause__, RuntimeError)
        assert history(tmp_path)[0]["status"] == "failed"

    def test_recovery_disabled_survives_a_broken_vram_probe(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        FlakyTrain(monkeypatch, exceptions=[oom()])
        t = make(tiny_dir, tmp_path, oom_recovery=False)
        with patch("backpropagate.gpu_safety.get_gpu_status", side_effect=RuntimeError("nvml")):
            with pytest.raises(GPUMemoryError) as ei:
                t.train(chat_jsonl, steps=1)
        assert "adjacent_marker" not in ei.value.details
        assert "vram_total_gb" not in ei.value.details

    def test_recovery_disabled_with_unavailable_gpu_snapshot_has_no_vram_detail(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        from backpropagate.gpu_safety import GPUStatus

        FlakyTrain(monkeypatch, exceptions=[oom()])
        t = make(tiny_dir, tmp_path, oom_recovery=False)
        with patch("backpropagate.gpu_safety.get_gpu_status", return_value=GPUStatus(available=False)):
            with pytest.raises(GPUMemoryError) as ei:
                t.train(chat_jsonl, steps=1)
        assert "vram_total_gb" not in ei.value.details

    def test_cache_reclaim_failure_does_not_block_recovery(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        armed = {"on": False}
        # CUDA "appears" available exactly once: at the recovery handler's cache reclaim.
        monkeypatch.setattr(torch.cuda, "is_available",
                            lambda: armed.pop("on", False) if "on" in armed else False)
        monkeypatch.setattr(torch.cuda, "empty_cache",
                            lambda: (_ for _ in ()).throw(RuntimeError("cuda context gone")))

        def arm():
            armed["on"] = True

        FlakyTrain(monkeypatch, exceptions=[oom()], before_raise=arm)
        t = make(tiny_dir, tmp_path, batch_size=2)
        run = t.train(chat_jsonl, steps=1)
        assert run.metadata["oom_retries"] == 1

    def test_persistent_oom_at_batch_one_is_exhausted_with_structured_code(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        FlakyTrain(monkeypatch, exceptions=[oom() for _ in range(5)])
        t = make(tiny_dir, tmp_path, batch_size=1)
        with pytest.raises(TrainingError) as ei:
            t.train(chat_jsonl, steps=1)
        assert ei.value.code == "RUNTIME_OOM_RECOVERY_EXHAUSTED"
        assert ei.value.details["consecutive_oom_at_min_batch"] == Trainer._OOM_MAX_RETRIES_AT_MIN_BATCH
        assert history(tmp_path)[0]["status"] == "failed"

    def test_transient_oom_at_batch_one_retries_same_settings_then_succeeds(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        flaky = FlakyTrain(monkeypatch, exceptions=[oom(), oom()])
        t = make(tiny_dir, tmp_path, batch_size=1)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            run = t.train(chat_jsonl, steps=1)
        assert flaky.calls == 3
        assert run.metadata["effective_batch_size"] == 1
        assert caplog.text.count("event=oom_recovery_at_floor") == 2


# ---------------------------------------------------------------------------
# Failure routing
# ---------------------------------------------------------------------------

class TestFailureRouting:
    def test_runtime_error_becomes_training_error_and_fires_on_error(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        errors = []
        FlakyTrain(monkeypatch, exceptions=[RuntimeError("loss is nan, aborting")])
        t = make(tiny_dir, tmp_path)
        with pytest.raises(TrainingError) as ei:
            t.train(chat_jsonl, steps=1, callback=TrainingCallback(on_error=errors.append))
        assert "Training failed: loss is nan, aborting" in str(ei.value)
        assert ei.value.suggestion and "--verbose" in ei.value.suggestion
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
        (row,) = history(tmp_path)
        assert row["status"] == "failed" and "loss is nan" in row["failure_reason"]

    def test_non_oom_cuda_runtime_error_gets_gpu_guidance(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        FlakyTrain(monkeypatch, exceptions=[RuntimeError("CUDA error: device-side assert triggered")])
        t = make(tiny_dir, tmp_path)
        with pytest.raises(TrainingError) as ei:
            t.train(chat_jsonl, steps=1)
        assert "GPU error during training" in str(ei.value)
        assert "reduce --batch-size" in ei.value.suggestion

    def test_unexpected_exception_type_is_wrapped_with_a_suggestion(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        errors = []
        FlakyTrain(monkeypatch, exceptions=[ValueError("bad collator")])
        t = make(tiny_dir, tmp_path)
        with pytest.raises(TrainingError) as ei:
            t.train(chat_jsonl, steps=1, callback=TrainingCallback(on_error=errors.append))
        assert "Training failed: bad collator" in str(ei.value)
        assert "unexpected exception escaped the training loop" in ei.value.suggestion
        assert isinstance(ei.value.__cause__, ValueError)
        assert errors and isinstance(errors[0], ValueError)

    def test_history_failures_during_error_handling_are_swallowed(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        FlakyTrain(monkeypatch, exceptions=[ValueError("x")])
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(TrainingError):
                t.train(chat_jsonl, steps=1)
        assert "RunHistoryManager.record_run_failed failed: ro" in caplog.text

    def test_runtime_error_history_failure_swallowed_without_callback(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        FlakyTrain(monkeypatch, exceptions=[RuntimeError("weird")])
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(TrainingError, match="Training failed: weird"):
                t.train(chat_jsonl, steps=1)
        assert "record_run_failed failed: ro" in caplog.text

    def test_structured_errors_propagate_unwrapped_after_history_update(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        original = TrainingError("custom structured failure", code="RUNTIME_TRAINING_FAILED")
        FlakyTrain(monkeypatch, exceptions=[original])
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(BackpropagateError) as ei:
                t.train(chat_jsonl, steps=1)
        assert ei.value is original
        assert "record_run_failed failed: ro" in caplog.text

    def test_keyboard_interrupt_aborts_with_structured_error_and_history(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch, caplog
    ):
        FlakyTrain(monkeypatch, exceptions=[KeyboardInterrupt()])
        t = make(tiny_dir, tmp_path)
        with patch.object(RunHistoryManager, "record_run_failed", side_effect=OSError("ro")), \
                caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(TrainingAbortedError) as ei:
                t.train(chat_jsonl, steps=1)
        assert "User interrupted training" in str(ei.value)
        assert "record_run_failed failed: ro" in caplog.text

    def test_batch_and_accumulation_are_restored_after_failure(
        self, tiny_dir, chat_jsonl, tmp_path, monkeypatch
    ):
        FlakyTrain(monkeypatch, exceptions=[oom(), ValueError("then something else")])
        t = make(tiny_dir, tmp_path, batch_size=4, gradient_accumulation=1)
        with pytest.raises(TrainingError):
            t.train(chat_jsonl, steps=1)
        assert (t.batch_size, t.gradient_accumulation) == (4, 1)
