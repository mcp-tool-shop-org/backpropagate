"""Coverage tests for ``cmd_train`` and ``cmd_multi_run`` in cli.py.

Mocked (real boundary: the long training loop / model load): ``Trainer`` and
``MultiRunTrainer`` are replaced by small recording fakes with real
``__init__`` signatures so the CLI's kwarg-introspection filter runs for real.
``MultiRunConfig`` and ``TrainingCallback`` are the real classes. Everything
else (argument parsing, error mapping, output formatting, exit codes) is real.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backpropagate import cli
from backpropagate.exceptions import (
    BackpropagateError,
    DatasetError,
    PartialSuccess,
    TrainingError,
    UserInputError,
)
from tests.helpers.cli_cov_support import parse

# ---------------------------------------------------------------------------
# cmd_train
# ---------------------------------------------------------------------------


def _make_fake_trainer(record: dict, *, train_exc=None, run_id=None, steps_reported=True):
    class FakeTrainer:
        def __init__(self, model, lora_r, learning_rate, batch_size, output_dir,
                     use_unsloth, use_dora=False, packing=True, method="sft",
                     full_ft_engine="default", switch_block_every=None,
                     block_order=None, block_writeback=None, block_train_embeddings=True,
                     simpo_beta=None):
            record["init"] = {
                "model": model, "lora_r": lora_r, "learning_rate": learning_rate, "batch_size": batch_size,
                "output_dir": output_dir, "use_unsloth": use_unsloth, "use_dora": use_dora,
                "packing": packing, "method": method, "full_ft_engine": full_ft_engine,
                "switch_block_every": switch_block_every, "block_order": block_order,
                "block_writeback": block_writeback, "block_train_embeddings": block_train_embeddings,
                "simpo_beta": simpo_beta,
            }

        def train(self, dataset, steps, samples, callback, resume_from):
            record["train"] = {"dataset": dataset, "steps": steps, "samples": samples, "resume": resume_from}
            if train_exc is not None:
                raise train_exc
            if steps_reported:
                callback.on_step(1, 0.5)
            return SimpleNamespace(final_loss=0.25, duration_seconds=3.5, run_id=run_id)

        def save(self, out):
            record["save"] = out
            return out

    return FakeTrainer


@pytest.fixture
def patch_trainer(monkeypatch):
    def _install(**kw):
        record: dict = {}
        monkeypatch.setattr("backpropagate.trainer.Trainer", _make_fake_trainer(record, **kw))
        return record

    return _install


class TestCmdTrainHappyPath:
    def test_success_prints_summary_and_forwards_filtered_kwargs(self, patch_trainer, capsys, tmp_path):
        record = patch_trainer(run_id="trainer-run-xyz")
        args = parse([
            "train", "--data", "d.jsonl", "--steps", "5", "--samples", "10",
            "--output", str(tmp_path), "--use-dora", "--no-packing",
            "--no-unsloth", "--resume", "ckpt", "--simpo-beta", "3.0",
        ])
        assert cli.cmd_train(args) == cli.EXIT_OK
        out = capsys.readouterr()
        assert "Samples: 10" in out.out
        assert "Training complete!" in out.out
        assert "Final loss: 0.2500" in out.out
        assert "Trainer run_id: trainer-run-xyz" in out.out
        assert "loss=0.5000" in out.out  # the on_step progress callback ran
        assert "Run ID:" in out.err
        init = record["init"]
        assert init["use_dora"] is True and init["packing"] is False
        assert init["use_unsloth"] is False
        assert init["simpo_beta"] == 3.0
        # kwargs the fake Trainer does not accept (fp8, backend...) were filtered, not passed
        assert record["train"]["resume"] == "ckpt"
        assert record["save"] == str(tmp_path)

    def test_block_engine_knobs_forwarded_only_with_block_engine(self, patch_trainer, capsys):
        record = patch_trainer()
        args = parse([
            "train", "--data", "d", "--full-ft-engine", "block", "--switch-block-every", "7",
            "--block-order", "ascending", "--block-writeback", "nearest",
            "--block-freeze-embeddings",
        ])
        assert cli.cmd_train(args) == cli.EXIT_OK
        init = record["init"]
        assert init["full_ft_engine"] == "block"
        assert init["switch_block_every"] == 7
        assert init["block_order"] == "ascending"
        assert init["block_writeback"] == "nearest"
        assert init["block_train_embeddings"] is False

    def test_default_engine_does_not_forward_block_knobs(self, patch_trainer):
        record = patch_trainer()
        assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_OK
        assert record["init"]["full_ft_engine"] == "default"
        assert record["init"]["switch_block_every"] is None

    def test_opaque_trainer_signature_degrades_to_no_extra_kwargs(self, monkeypatch, capsys):
        """A Trainer whose signature cannot be introspected gets only the legacy kwargs."""
        seen: dict = {}

        class Opaque:
            def __init__(self, **kw):  # replaced below by a non-introspectable callable
                seen["kw"] = kw

            def train(self, **kw):
                return SimpleNamespace(final_loss=1.0, duration_seconds=1.0)

            def save(self, out):
                return out

        # inspect.signature(Trainer.__init__) raising ValueError -> set() filter.
        import inspect

        real_sig = inspect.signature

        def fake_sig(obj, *a, **k):
            if obj is Opaque.__init__:
                raise ValueError("no signature")
            return real_sig(obj, *a, **k)

        monkeypatch.setattr("backpropagate.trainer.Trainer", Opaque)
        monkeypatch.setattr(inspect, "signature", fake_sig)
        assert cli.cmd_train(parse(["train", "--data", "d", "--use-dora"])) == cli.EXIT_OK
        assert "use_dora" not in seen["kw"]
        # No --lora-r: the CLI passes None and the trainer resolves the rank
        # (the LoRA preset that fits the GPU, or the settings value).
        assert seen["kw"]["lora_r"] is None

    def test_var_keyword_trainer_gets_everything(self, monkeypatch):
        seen: dict = {}

        class Wide:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def train(self, **kw):
                return SimpleNamespace(final_loss=1.0, duration_seconds=1.0)

            def save(self, out):
                return out

        monkeypatch.setattr("backpropagate.trainer.Trainer", Wide)
        assert cli.cmd_train(parse(["train", "--data", "d", "--fp8", "--backend", "cuda"])) == cli.EXIT_OK
        assert seen["fp8"] is True and seen["backend"] == "cuda"
        assert "simpo_beta" not in seen  # unset optional hyperparameters are dropped, not None-forwarded

    def test_observability_failures_do_not_abort(self, patch_trainer, monkeypatch, capsys):
        patch_trainer()

        def boom(*a, **k):
            raise RuntimeError("log backend down")

        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_OK
        assert "Training complete!" in capsys.readouterr().out

    def test_no_data_prints_hint(self, capsys):
        assert cli.cmd_train(parse(["train"])) == cli.EXIT_USER_ERROR
        err = capsys.readouterr()
        assert "No dataset specified." in err.err
        assert "backprop train --data my_data.jsonl" in err.out


class TestCmdTrainErrorMapping:
    @pytest.mark.parametrize(
        "exc, code, label",
        [
            (UserInputError("bad input", hint="fix it"), cli.EXIT_USER_ERROR, "bad input"),
            (DatasetError("bad data", suggestion="fix it"), cli.EXIT_USER_ERROR, "Dataset error: bad data"),
            (TrainingError("oom", suggestion="fix it"), cli.EXIT_RUNTIME_ERROR, "Training error: oom"),
            (BackpropagateError("other", suggestion="fix it"), cli.EXIT_RUNTIME_ERROR, "other"),
        ],
    )
    @pytest.mark.parametrize("verbose", [False, True])
    def test_structured_errors(self, patch_trainer, capsys, exc, code, label, verbose):
        patch_trainer(train_exc=exc)
        argv = ["train", "--data", "d"]
        args = parse(argv)
        args.verbose = verbose
        assert cli.cmd_train(args) == code
        captured = capsys.readouterr()
        assert label in captured.err
        assert "Suggestion: fix it" in captured.out

    def test_partial_success(self, patch_trainer, capsys):
        patch_trainer(train_exc=PartialSuccess("half done", total_items=2, succeeded=1, failed=1, suggestion="retry"))
        assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_PARTIAL_SUCCESS
        out = capsys.readouterr().out
        assert "half done" in out and "Suggestion: retry" in out

    def test_keyboard_interrupt(self, patch_trainer, capsys):
        patch_trainer(train_exc=KeyboardInterrupt())
        assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_INTERRUPTED
        assert "interrupted by user" in capsys.readouterr().out

    def test_unexpected_exception_is_redacted(self, patch_trainer, capsys):
        patch_trainer(train_exc=RuntimeError("Authorization: Bearer abcdef1234567890"))
        assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "Training failed: RuntimeError" in captured.err
        assert "abcdef1234567890" not in captured.err
        assert "Run with --verbose" in captured.out

    def test_unexpected_exception_verbose_prints_traceback(self, patch_trainer, capsys):
        patch_trainer(train_exc=RuntimeError("kaboom"))
        args = parse(["train", "--data", "d"])
        args.verbose = True
        assert cli.cmd_train(args) == cli.EXIT_RUNTIME_ERROR
        err = capsys.readouterr().err
        assert "Training failed: kaboom" in err
        assert "Traceback" in err


# ---------------------------------------------------------------------------
# cmd_multi_run
# ---------------------------------------------------------------------------


def _make_fake_multi(record: dict, *, run_exc=None, failed_runs=0, call_hook=True, ctor_exc=None):
    class FakeMulti:
        def __init__(self, model, config, on_run_complete, resume_from=None, use_dora=False,
                     **extra):
            if ctor_exc is not None:
                raise ctor_exc
            record["init"] = {"model": model, "config": config, "resume_from": resume_from,
                                  "use_dora": use_dora, "extra": extra}
            self._cb = on_run_complete

        def run(self, data):
            record["data"] = data
            if run_exc is not None:
                raise run_exc
            if call_hook:
                self._cb(SimpleNamespace(run_index=0, final_loss=0.75))
            return SimpleNamespace(
                total_runs=3, final_loss=0.5, total_duration_seconds=12.0,
                final_checkpoint_path=None, failed_runs=failed_runs,
            )

    return FakeMulti


@pytest.fixture
def patch_multi(monkeypatch):
    def _install(**kw):
        record: dict = {}
        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _make_fake_multi(record, **kw))
        return record

    return _install


class TestCmdMultiRun:
    def test_success_with_gates_and_strategy(self, patch_multi, capsys, tmp_path):
        record = patch_multi()
        args = parse([
            "multi-run", "--data", "d.jsonl", "--runs", "3", "--steps", "4", "--samples", "9",
            "--output", str(tmp_path), "--use-dora", "--resume", "prev",
            "--drift-gate", "--eval-gate",
        ])
        assert cli.cmd_multi_run(args) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Run 1 complete: loss=0.7500" in out
        assert "Multi-run training complete!" in out
        assert "Total runs: 3" in out
        assert "drift=on" in out and "eval=on" in out
        cfg = record["init"]["config"]
        assert cfg.num_runs == 3 and cfg.steps_per_run == 4 and cfg.samples_per_run == 9
        assert cfg.drift_gate is True and cfg.eval_gate is True
        assert cfg.use_dora is True
        assert record["init"]["extra"]["method"] == "sft"
        assert record["init"]["resume_from"] == "prev"
        assert record["data"] == "d.jsonl"

    def test_gates_off_banner(self, patch_multi, capsys):
        patch_multi()
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_OK
        assert "drift=off, eval=off" in capsys.readouterr().out

    def test_partial_success_when_runs_failed(self, patch_multi, capsys):
        patch_multi(failed_runs=2)
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_PARTIAL_SUCCESS
        out = capsys.readouterr().out
        assert "2/3 runs failed (partial success)" in out
        assert "backprop runs --status failed" in out

    def test_no_data(self, capsys):
        assert cli.cmd_multi_run(parse(["multi-run"])) == cli.EXIT_USER_ERROR
        assert "No dataset specified." in capsys.readouterr().err

    def test_observability_failures_do_not_abort(self, patch_multi, monkeypatch):
        patch_multi()

        def boom(*a, **k):
            raise RuntimeError("log backend down")

        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_OK

    def test_opaque_signatures_degrade(self, monkeypatch):
        """Un-introspectable MultiRunConfig/MultiRunTrainer get only the baseline kwargs."""
        import inspect

        seen: dict = {}

        class Opaque:
            def __init__(self, model, config, on_run_complete, resume_from=None):
                seen["ok"] = True

            def run(self, data):
                return SimpleNamespace(total_runs=1, final_loss=0.1, total_duration_seconds=1.0,
                                       final_checkpoint_path="ckpt", failed_runs=0)

        real_sig = inspect.signature

        def fake_sig(obj, *a, **k):
            if obj is Opaque.__init__:
                raise TypeError("opaque")
            return real_sig(obj, *a, **k)

        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", Opaque)
        monkeypatch.setattr(inspect, "signature", fake_sig)

        class NotADataclass:
            def __init__(self, **kw):
                self.kw = kw

        monkeypatch.setattr("backpropagate.multi_run.MultiRunConfig", NotADataclass)
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_OK
        assert seen["ok"]

    def test_multi_trainer_without_var_keyword_only_gets_accepted_kwargs(self, monkeypatch):
        """Candidate kwargs the installed MultiRunTrainer does not accept are filtered, not passed."""
        seen = {}

        class Strict:
            def __init__(self, model, config, on_run_complete, resume_from=None):
                seen["ok"] = True

            def run(self, data):
                return SimpleNamespace(total_runs=1, final_loss=0.1, total_duration_seconds=1.0,
                                       final_checkpoint_path="ckpt", failed_runs=0)

        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", Strict)
        # --method stays sft: a non-SFT method is refused outright when the trainer
        # takes no `method` (#254). --simpo-beta still has to be filtered out.
        argv = ["multi-run", "--data", "d", "--method", "sft", "--simpo-beta", "2.5"]
        assert cli.cmd_multi_run(parse(argv)) == cli.EXIT_OK  # a TypeError here would mean nothing was filtered
        assert seen["ok"]

    def test_var_keyword_multi_trainer_receives_non_config_kwargs(self, monkeypatch):
        seen: dict = {}

        class Wide:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def run(self, data):
                return SimpleNamespace(total_runs=1, final_loss=0.1, total_duration_seconds=1.0,
                                       final_checkpoint_path="ckpt", failed_runs=0)

        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", Wide)
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d", "--method", "orpo", "--orpo-beta", "0.3"])) == cli.EXIT_OK
        assert seen["method"] == "orpo" and seen["orpo_beta"] == 0.3

    @pytest.mark.parametrize(
        "exc, code, err_text",
        [
            (UserInputError("bad input", hint="fix it"), cli.EXIT_USER_ERROR, "bad input"),
            (DatasetError("bad data", suggestion="fix it"), cli.EXIT_USER_ERROR, "Dataset error: bad data"),
            (BackpropagateError("other", suggestion="fix it"), cli.EXIT_RUNTIME_ERROR, "other"),
        ],
    )
    @pytest.mark.parametrize("verbose", [False, True])
    def test_structured_errors(self, patch_multi, capsys, exc, code, err_text, verbose):
        patch_multi(run_exc=exc)
        args = parse(["multi-run", "--data", "d"])
        args.verbose = verbose
        assert cli.cmd_multi_run(args) == code
        captured = capsys.readouterr()
        assert err_text in captured.err
        assert "Suggestion: fix it" in captured.out

    def test_partial_success_exception(self, patch_multi, capsys):
        patch_multi(run_exc=PartialSuccess("some failed", total_items=2, succeeded=1, failed=1, suggestion="resume"))
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_PARTIAL_SUCCESS
        out = capsys.readouterr().out
        assert "some failed" in out and "Suggestion: resume" in out

    def test_keyboard_interrupt(self, patch_multi, capsys):
        patch_multi(run_exc=KeyboardInterrupt())
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_INTERRUPTED

    def test_value_error_is_user_error(self, patch_multi, capsys):
        patch_multi(run_exc=ValueError("bad merge mode"))
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_USER_ERROR
        assert "Invalid argument: bad merge mode" in capsys.readouterr().err

    def test_unexpected_exception_redacted_and_verbose(self, patch_multi, capsys):
        patch_multi(run_exc=RuntimeError("Authorization: Bearer abcdef1234567890"))
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err
        assert "Run with --verbose" in captured.out

        args = parse(["multi-run", "--data", "d"])
        args.verbose = True
        assert cli.cmd_multi_run(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err
