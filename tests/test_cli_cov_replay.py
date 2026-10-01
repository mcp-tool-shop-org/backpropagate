"""Coverage tests for ``cmd_replay``, ``_parse_replay_override`` and
``cmd_export_runs`` in cli.py.

Real: the run history (real ``RunHistoryManager`` on ``tmp_path``), override
parsing/coercion, JSONL files written to disk, exit codes, output.
Mocked (real boundary: model load + training loop): ``Trainer`` and
``MultiRunTrainer`` are small recording fakes.
"""

from __future__ import annotations

import argparse
import inspect
import json
from types import SimpleNamespace

import pytest

from backpropagate import cli
from backpropagate.exceptions import BackpropagateError
from tests.helpers.cli_cov_support import last_json, parse, seed_runs


@pytest.fixture
def hist(tmp_path):
    out = tmp_path / "out"
    seed_runs(out, [
        {
            "run_id": "single-aaaa", "status": "completed", "model_name": "tiny/model",
            "dataset_info": "data.jsonl", "session_kind": "single_run",
            "hyperparameters": {"lora_r": 8, "learning_rate": 1e-4, "max_steps": 6, "samples": 20,
                                "batch_size": 2, "gradient_accumulation": 4, "lora_alpha": 16,
                                "lora_dropout": 0.05},
        },
        {
            "run_id": "multi-bbbb", "status": "completed", "model_name": "tiny/model",
            "dataset_info": "multi.jsonl", "session_kind": "multi_run",
            "hyperparameters": {"num_runs": 2, "steps_per_run": 3, "samples_per_run": 5,
                                "merge_mode": "simple", "use_dora": True},
        },
        {"run_id": "nodata-cccc", "status": "failed", "model_name": "tiny/model"},
    ])
    return out


class _Rec:
    pass


@pytest.fixture
def fakes(monkeypatch):
    rec = _Rec()
    rec.single_init = None
    rec.single_train = None
    rec.multi_init = None
    rec.multi_run_data = None
    rec.exc = None

    class FakeTrainer:
        def __init__(self, model, lora_r, learning_rate, output_dir, batch_size=None,
                     gradient_accumulation=None, lora_alpha=None, lora_dropout=None, use_dora=False):
            rec.single_init = {
                "model": model, "lora_r": lora_r, "learning_rate": learning_rate, "output_dir": output_dir,
                "batch_size": batch_size, "gradient_accumulation": gradient_accumulation,
                "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "use_dora": use_dora,
            }

        def train(self, dataset, steps, samples, callback):
            if rec.exc is not None:
                raise rec.exc
            rec.single_train = {"dataset": dataset, "steps": steps, "samples": samples}
            return SimpleNamespace(final_loss=0.2, run_id="new-run-123")

        def save(self, out):
            rec.saved = out

    class FakeMulti:
        def __init__(self, model, config, use_dora=False):
            rec.multi_init = {"model": model, "config": config, "use_dora": use_dora}

        def run(self, dataset):
            if rec.exc is not None:
                raise rec.exc
            rec.multi_run_data = dataset
            return SimpleNamespace(total_runs=2, final_loss=0.3)

    monkeypatch.setattr("backpropagate.trainer.Trainer", FakeTrainer)
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", FakeMulti)
    return rec


class TestParseOverride:
    def test_valid(self):
        assert cli._parse_replay_override("lr=0.1") == ("lr", "0.1")
        assert cli._parse_replay_override(" lr = x=y") == ("lr", " x=y")

    def test_missing_equals(self):
        with pytest.raises(argparse.ArgumentTypeError, match="no '=' separator"):
            cli._parse_replay_override("lr")

    def test_empty_key(self):
        with pytest.raises(argparse.ArgumentTypeError, match="key is empty"):
            cli._parse_replay_override("  =3")


class TestReplayPreflight:
    def test_missing_history_dir(self, tmp_path, capsys):
        assert cli.cmd_replay(parse(["replay", "x", "--output", str(tmp_path / "no")])) == cli.EXIT_USER_ERROR
        assert "No history directory" in capsys.readouterr().err

    def test_unknown_run(self, hist, capsys):
        assert cli.cmd_replay(parse(["replay", "zzz", "--output", str(hist)])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "run_id" in captured.err
        assert "Next steps" in captured.out

    def test_override_key_not_whitelisted(self, hist, capsys):
        args = parse(["replay", "single-aaaa", "--output", str(hist), "--override", "lr_rate=1"])
        assert cli.cmd_replay(args) == cli.EXIT_USER_ERROR
        assert "'lr_rate' is not in the allowed set" in capsys.readouterr().err

    def test_run_without_dataset(self, hist, capsys):
        assert cli.cmd_replay(parse(["replay", "nodata-cccc", "--output", str(hist)])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "has no dataset_info" in captured.err
        assert "backprop train --data" in captured.out

    @pytest.mark.parametrize("bad", ["batch_size=foo", "lora_r=abc"])
    def test_non_numeric_override_for_numeric_key(self, hist, capsys, bad):
        args = parse(["replay", "single-aaaa", "--output", str(hist), "--override", bad])
        assert cli.cmd_replay(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "is not numeric" in captured.out and "Suggestion:" in captured.out

    def test_observability_failures_are_ignored(self, hist, fakes, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        assert cli.cmd_replay(parse(["replay", "single-aaaa", "--output", str(hist)])) == cli.EXIT_OK


class TestReplaySingleRun:
    def test_inherits_recorded_hyperparameters(self, hist, fakes, capsys):
        assert cli.cmd_replay(parse(["replay", "single-aaaa", "--output", str(hist)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Replay complete!" in out and "Final loss: 0.2000" in out
        assert "backprop diff-runs single-aaaa <new-run-id>" in out
        assert fakes.single_init == {
            "model": "tiny/model", "lora_r": 8, "learning_rate": 1e-4, "output_dir": str(hist),
            "batch_size": 2, "gradient_accumulation": 4, "lora_alpha": 16, "lora_dropout": 0.05,
            "use_dora": False,
        }
        assert fakes.single_train == {"dataset": "data.jsonl", "steps": 6, "samples": 20}
        assert fakes.saved == str(hist)

    def test_overrides_are_coerced(self, hist, fakes, capsys):
        argv = ["replay", "single-aaaa", "--output", str(hist),
                "--override", "learning_rate=0.5", "--override", "lora_r=32",
                "--override", "batch_size=auto", "--override", "use_dora=TRUE", "--override", "optim=adamw"]
        assert cli.cmd_replay(parse(argv)) == cli.EXIT_OK
        init = fakes.single_init
        assert init["learning_rate"] == 0.5 and init["lora_r"] == 32
        assert init["batch_size"] == "auto" and init["use_dora"] is True
        assert "Overrides:" in capsys.readouterr().out

    def test_json_payload(self, hist, fakes, capsys):
        argv = ["replay", "single-aaaa", "--output", str(hist), "--json", "--override", "lora_r=4"]
        assert cli.cmd_replay(parse(argv)) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["session_kind"] == "single_run"
        assert payload["original_run_id"] == "single-aaaa"
        assert payload["new_run_id"] == "new-run-123"
        assert payload["overrides"] == {"lora_r": "4"}
        assert payload["final_loss"] == 0.2

    def test_var_keyword_trainer_gets_every_candidate(self, hist, monkeypatch):
        seen = {}

        class Wide:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def train(self, **kw):
                return SimpleNamespace(final_loss=0.1)

            def save(self, out):
                pass

        monkeypatch.setattr("backpropagate.trainer.Trainer", Wide)
        argv = ["replay", "single-aaaa", "--output", str(hist), "--override", "optim=sgd"]
        assert cli.cmd_replay(parse(argv)) == cli.EXIT_OK
        assert seen["optim"] == "sgd" and seen["lora_alpha"] == 16

    def test_unintrospectable_trainer_signature_is_tolerated(self, hist, fakes, monkeypatch):
        real_sig = inspect.signature

        def fake_sig(obj, *a, **k):
            if getattr(obj, "__qualname__", "").endswith("FakeTrainer.__init__"):
                raise ValueError("no signature")
            return real_sig(obj, *a, **k)

        monkeypatch.setattr(inspect, "signature", fake_sig)
        assert cli.cmd_replay(parse(["replay", "single-aaaa", "--output", str(hist)])) == cli.EXIT_OK

    def test_json_without_new_run_id_or_numeric_loss(self, hist, monkeypatch, capsys):
        class T:
            def __init__(self, **kw):
                pass

            def train(self, **kw):
                return SimpleNamespace(final_loss="n/a")

            def save(self, out):
                pass

        monkeypatch.setattr("backpropagate.trainer.Trainer", T)
        assert cli.cmd_replay(parse(["replay", "single-aaaa", "--output", str(hist), "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["new_run_id"] is None and payload["final_loss"] is None


class TestReplayMultiRun:
    def test_human(self, hist, fakes, capsys):
        assert cli.cmd_replay(parse(["replay", "multi-bbbb", "--output", str(hist)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Replay (multi-run) complete!" in out and "Total runs: 2" in out
        cfg = fakes.multi_init["config"]
        assert (cfg.num_runs, cfg.steps_per_run, cfg.samples_per_run) == (2, 3, 5)
        assert cfg.merge_mode.value == "simple"
        assert cfg.use_dora is True  # Wave-6b key routed to the config field it lives on
        assert fakes.multi_run_data == "multi.jsonl"

    def test_json(self, hist, fakes, capsys):
        assert cli.cmd_replay(parse(["replay", "multi-bbbb", "--output", str(hist), "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["session_kind"] == "multi_run"
        assert payload["total_runs"] == 2 and payload["final_loss"] == 0.3

    def test_json_non_numeric_loss(self, hist, monkeypatch, capsys):
        class M:
            def __init__(self, **kw):
                pass

            def run(self, dataset):
                return SimpleNamespace(total_runs=1, final_loss=None)

        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", M)
        assert cli.cmd_replay(parse(["replay", "multi-bbbb", "--output", str(hist), "--json"])) == cli.EXIT_OK
        assert last_json(capsys.readouterr().out)["final_loss"] is None

    def test_opaque_signatures_degrade(self, hist, monkeypatch):
        """A non-dataclass config and an un-introspectable trainer get only baseline kwargs."""
        seen = {}

        class Cfg:
            def __init__(self, **kw):
                seen["cfg"] = kw

        class M:
            def __init__(self, model, config):
                seen["trainer"] = True

            def run(self, dataset):
                return SimpleNamespace(total_runs=1, final_loss=0.5)

        real_sig = inspect.signature

        def fake_sig(obj, *a, **k):
            if getattr(obj, "__qualname__", "").endswith("M.__init__"):
                raise TypeError("opaque")
            return real_sig(obj, *a, **k)

        monkeypatch.setattr("backpropagate.multi_run.MultiRunConfig", Cfg)
        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", M)
        monkeypatch.setattr(inspect, "signature", fake_sig)
        assert cli.cmd_replay(parse(["replay", "multi-bbbb", "--output", str(hist)])) == cli.EXIT_OK
        assert "use_dora" not in seen["cfg"] and seen["trainer"]

    def test_var_keyword_multi_trainer(self, hist, monkeypatch):
        seen = {}

        class Cfg:
            def __init__(self, **kw):
                pass

        class M:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def run(self, dataset):
                return SimpleNamespace(total_runs=1, final_loss=0.5)

        monkeypatch.setattr("backpropagate.multi_run.MultiRunConfig", Cfg)
        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", M)
        assert cli.cmd_replay(parse(["replay", "multi-bbbb", "--output", str(hist)])) == cli.EXIT_OK
        assert seen["use_dora"] is True


class TestReplayErrors:
    def test_keyboard_interrupt(self, hist, fakes, capsys):
        fakes.exc = KeyboardInterrupt()
        assert cli.cmd_replay(parse(["replay", "single-aaaa", "--output", str(hist)])) == cli.EXIT_INTERRUPTED
        assert "Replay interrupted by user" in capsys.readouterr().out

    def test_structured_error(self, hist, fakes, capsys):
        fakes.exc = BackpropagateError("oom", suggestion="smaller batch")
        assert cli.cmd_replay(parse(["replay", "single-aaaa", "--output", str(hist)])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "Replay failed: oom" in captured.err and "Suggestion: smaller batch" in captured.out

    def test_unexpected_error_redacted_then_verbose(self, hist, fakes, capsys):
        fakes.exc = RuntimeError("Authorization: Bearer abcdef1234567890")
        argv = ["replay", "single-aaaa", "--output", str(hist)]
        assert cli.cmd_replay(parse(argv)) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err and "Run with --verbose" in captured.out
        args = parse(argv)
        args.verbose = True
        assert cli.cmd_replay(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err


class TestExportRuns:
    def test_missing_dir(self, tmp_path, capsys):
        assert cli.cmd_export_runs(parse(["export-runs", "--output", str(tmp_path / "no")])) == cli.EXIT_USER_ERROR
        assert "No history directory" in capsys.readouterr().err

    def test_invalid_status(self, hist, capsys):
        args = parse(["export-runs", "--output", str(hist)])
        args.status = "bogus"
        assert cli.cmd_export_runs(args) == cli.EXIT_USER_ERROR
        assert "Invalid status 'bogus'" in capsys.readouterr().err

    def test_unsupported_format(self, hist, capsys):
        args = parse(["export-runs", "--output", str(hist)])
        args.format = "csv"
        assert cli.cmd_export_runs(args) == cli.EXIT_USER_ERROR
        assert "Unsupported --format 'csv'" in capsys.readouterr().err

    def test_stdout_jsonl_and_stderr_banner(self, hist, capsys):
        assert cli.cmd_export_runs(parse(["export-runs", "--output", str(hist)])) == cli.EXIT_OK
        captured = capsys.readouterr()
        lines = [json.loads(line) for line in captured.out.splitlines() if line.startswith("{")]
        assert {rec["run_id"] for rec in lines} == {"single-aaaa", "multi-bbbb", "nodata-cccc"}
        assert all(rec["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION for rec in lines)
        assert "Exported 3 run(s)" in captured.err

    def test_write_to_file_with_status_filter(self, hist, tmp_path, capsys):
        target = tmp_path / "nested" / "dump.jsonl"
        args = parse(["export-runs", "--output", str(hist), "--to", str(target), "--status", "failed"])
        assert cli.cmd_export_runs(args) == cli.EXIT_OK
        assert "Exported 1 run(s)" in capsys.readouterr().out
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        assert [r["run_id"] for r in records] == ["nodata-cccc"]

    def test_write_failure(self, hist, tmp_path, capsys):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        args = parse(["export-runs", "--output", str(hist), "--to", str(blocker / "dump.jsonl")])
        assert cli.cmd_export_runs(args) == cli.EXIT_RUNTIME_ERROR
        assert "Write failed" in capsys.readouterr().err
