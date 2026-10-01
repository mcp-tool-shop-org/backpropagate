"""Coverage tests for ``cmd_config``, ``cmd_resume``, ``cmd_push`` and the
timestamp helper in cli.py.

Real: the run history on disk (written through the real ``RunHistoryManager``),
settings, local-path handling, token-file resolution, exit codes and output.
Mocked (real boundaries): ``Trainer`` / ``MultiRunTrainer`` (model load +
training loop) and ``backpropagate.export.push_to_hub`` (Hugging Face Hub
network upload).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backpropagate import cli
from backpropagate.exceptions import BackpropagateError, ExportError, UserInputError
from tests.helpers.cli_cov_support import parse, seed_runs

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


class TestCmdConfig:
    def test_show_prints_sections(self, capsys):
        assert cli.cmd_config(parse(["config"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        for section in ("Model", "LoRA", "Training", "Data"):
            assert section in out
        assert "target_modules" in out and "output_dir" in out

    def test_windows_section_only_on_nt(self, monkeypatch, capsys):
        class NtOs:
            name = "nt"

            def __getattr__(self, item):
                import os

                return getattr(os, item)

        monkeypatch.setattr(cli, "os", NtOs())
        assert cli.cmd_config(parse(["config"])) == cli.EXIT_OK
        assert "pre_tokenize" in capsys.readouterr().out

        class PosixOs(NtOs):
            name = "posix"

        monkeypatch.setattr(cli, "os", PosixOs())
        assert cli.cmd_config(parse(["config"])) == cli.EXIT_OK
        assert "pre_tokenize" not in capsys.readouterr().out

    def test_reset_and_set_are_not_implemented(self, capsys):
        assert cli.cmd_config(parse(["config", "--reset"])) == cli.EXIT_USER_ERROR
        assert "reset via CLI is not implemented" in capsys.readouterr().err
        assert cli.cmd_config(parse(["config", "--set", "a=b"])) == cli.EXIT_USER_ERROR
        assert "--set is not implemented" in capsys.readouterr().err

    def test_keyboard_interrupt(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "_print_header", boom)
        assert cli.cmd_config(parse(["config"])) == cli.EXIT_INTERRUPTED
        assert "interrupted by user" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@pytest.fixture
def history(tmp_path):
    out = tmp_path / "out"
    seed_runs(out, [
        {
            "run_id": "single-0001", "status": "completed", "model_name": "tiny/model",
            "dataset_info": "data.jsonl", "session_kind": "single_run",
            "hyperparameters": {"lora_r": 8, "learning_rate": 1e-4, "max_steps": 7, "max_samples": 11},
        },
        {
            "run_id": "multi-0002", "status": "failed", "model_name": "tiny/model",
            "dataset_info": "multi.jsonl", "session_kind": "multi_run",
            "hyperparameters": {"num_runs": 2, "steps_per_run": 3, "samples_per_run": 4, "merge_mode": "simple"},
        },
    ])
    return out


class _Rec:
    pass


@pytest.fixture
def fake_trainers(monkeypatch):
    rec = _Rec()
    rec.single = {}
    rec.multi = {}
    rec.exc = None

    class FakeTrainer:
        def __init__(self, model, lora_r, learning_rate, output_dir):
            rec.single["init"] = {"model": model, "lora_r": lora_r, "lr": learning_rate, "out": output_dir}

        def train(self, dataset, steps, samples, callback, resume_from):
            if rec.exc is not None:
                raise rec.exc
            rec.single["train"] = {"dataset": dataset, "steps": steps, "samples": samples, "resume": resume_from}
            return SimpleNamespace(final_loss=0.125)

        def save(self, out):
            rec.single["save"] = out

    class FakeMulti:
        def __init__(self, model, config, resume_from):
            rec.multi["init"] = {"model": model, "config": config, "resume": resume_from}

        def run(self, dataset):
            if rec.exc is not None:
                raise rec.exc
            rec.multi["dataset"] = dataset
            return SimpleNamespace(total_runs=2, final_loss=0.375)

    monkeypatch.setattr("backpropagate.trainer.Trainer", FakeTrainer)
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", FakeMulti)
    return rec


class TestCmdResume:
    def test_missing_history_dir(self, tmp_path, capsys):
        args = parse(["resume", "x", "--output", str(tmp_path / "nowhere")])
        assert cli.cmd_resume(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "No history directory" in captured.err
        assert "--output <dir>" in captured.out

    def test_unknown_run_id(self, history, capsys):
        assert cli.cmd_resume(parse(["resume", "zzz", "--output", str(history)])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "No run matching 'zzz'" in captured.err
        assert "backprop runs --output" in captured.out

    def test_single_run_resume_uses_recorded_hyperparameters(self, history, fake_trainers, capsys):
        args = parse(["resume", "single-0001", "--output", str(history)])
        assert cli.cmd_resume(args) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "already marked as completed" in out
        assert "Resume complete!" in out and "Final loss: 0.1250" in out
        assert fake_trainers.single["init"] == {"model": "tiny/model", "lora_r": 8, "lr": 1e-4, "out": str(history)}
        assert fake_trainers.single["train"] == {
            "dataset": "data.jsonl", "steps": 7, "samples": 11, "resume": "single-0001",
        }
        assert fake_trainers.single["save"] == str(history)

    def test_data_flag_overrides_recorded_dataset(self, history, fake_trainers):
        args = parse(["resume", "single-0001", "--output", str(history), "--data", "other.jsonl"])
        assert cli.cmd_resume(args) == cli.EXIT_OK
        assert fake_trainers.single["train"]["dataset"] == "other.jsonl"

    def test_multi_run_resume(self, history, fake_trainers, capsys):
        assert cli.cmd_resume(parse(["resume", "multi-0002", "--output", str(history)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Multi-run resume complete!" in out and "Total runs: 2" in out
        assert "already marked as completed" not in out  # status was failed
        cfg = fake_trainers.multi["init"]["config"]
        assert (cfg.num_runs, cfg.steps_per_run, cfg.samples_per_run) == (2, 3, 4)
        assert cfg.merge_mode.value == "simple"
        assert fake_trainers.multi["init"]["resume"] == "multi-0002"
        assert fake_trainers.multi["dataset"] == "multi.jsonl"

    def test_keyboard_interrupt(self, history, fake_trainers, capsys):
        fake_trainers.exc = KeyboardInterrupt()
        assert cli.cmd_resume(parse(["resume", "single-0001", "--output", str(history)])) == cli.EXIT_INTERRUPTED
        assert "Resume interrupted by user" in capsys.readouterr().out

    def test_structured_error(self, history, fake_trainers, capsys):
        fake_trainers.exc = BackpropagateError("checkpoint corrupt", suggestion="restore from backup")
        assert cli.cmd_resume(parse(["resume", "multi-0002", "--output", str(history)])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "Resume failed: checkpoint corrupt" in captured.err
        assert "Suggestion: restore from backup" in captured.out

    def test_unexpected_error_redacted_and_verbose(self, history, fake_trainers, capsys):
        fake_trainers.exc = RuntimeError("Authorization: Bearer abcdef1234567890")
        argv = ["resume", "single-0001", "--output", str(history)]
        assert cli.cmd_resume(parse(argv)) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err
        assert "Run with --verbose" in captured.out
        args = parse(argv)
        args.verbose = True
        assert cli.cmd_resume(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err

    def test_observability_failures_ignored(self, history, fake_trainers, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr("backpropagate.logging_config.bind_run_context", boom)
        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        assert cli.cmd_resume(parse(["resume", "single-0001", "--output", str(history)])) == cli.EXIT_OK


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


@pytest.fixture
def push_env(monkeypatch, tmp_path):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "BACKPROPAGATE_QUIET_TOKEN_HINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli, "_ENV_TOKEN_CALIBRATION_WARNED", False)
    local = tmp_path / "export"
    local.mkdir()
    rec = _Rec()
    rec.calls = []
    rec.exc = None

    def fake_push(**kw):
        rec.calls.append(kw)
        if rec.exc is not None:
            raise rec.exc
        return "https://huggingface.co/me/model"

    monkeypatch.setattr("backpropagate.export.push_to_hub", fake_push)
    rec.local = local
    return rec


class TestCmdPush:
    def test_missing_local_path(self, tmp_path, capsys):
        assert cli.cmd_push(parse(["push", str(tmp_path / "no"), "--repo", "a/b"])) == cli.EXIT_USER_ERROR
        assert "Local path does not exist" in capsys.readouterr().err

    def test_missing_repo_direct_call(self, push_env, capsys):
        args = parse(["push", str(push_env.local), "--repo", "a/b"])
        args.repo = None
        assert cli.cmd_push(args) == cli.EXIT_USER_ERROR
        assert "No target repo specified." in capsys.readouterr().err

    def test_success_forwards_options(self, push_env, capsys):
        argv = ["push", str(push_env.local), "--repo", "me/model", "--private", "--include-base",
                "--hub-revision", "dev", "--hub-commit-message", "msg"]
        assert cli.cmd_push(parse(argv)) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Visibility: private" in out
        assert "Including base model files" in out
        assert "Pushed to Hub: https://huggingface.co/me/model" in out
        call = push_env.calls[0]
        assert call["repo_id"] == "me/model" and call["private"] is True and call["include_base"] is True
        assert call["revision"] == "dev" and call["commit_message"] == "msg"
        assert call["token"] is None

    def test_inline_token_warns_and_is_forwarded(self, push_env, capsys):
        argv = ["push", str(push_env.local), "--repo", "me/model", "--token", "hf_inline"]
        assert cli.cmd_push(parse(argv)) == cli.EXIT_OK
        assert "--token was passed on the command line" in capsys.readouterr().out
        assert push_env.calls[0]["token"] == "hf_inline"

    def test_token_file_is_read(self, push_env, tmp_path):
        tok = tmp_path / "tok"
        tok.write_text("hf_fromfile\n", encoding="utf-8")
        argv = ["push", str(push_env.local), "--repo", "me/model", "--token-file", str(tok)]
        assert cli.cmd_push(parse(argv)) == cli.EXIT_OK
        assert push_env.calls[0]["token"] == "hf_fromfile"

    def test_missing_token_file_is_user_error(self, push_env, tmp_path, capsys):
        argv = ["push", str(push_env.local), "--repo", "me/model", "--token-file", str(tmp_path / "nope")]
        assert cli.cmd_push(parse(argv)) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "--token-file path does not exist" in captured.err
        assert "Suggestion:" not in captured.err and "chmod 600" in captured.out
        assert push_env.calls == []

    def test_token_and_token_file_mutex(self, push_env):
        argv = ["push", str(push_env.local), "--repo", "me/model", "--token", "t", "--token-file", "f"]
        with pytest.raises(UserInputError, match="mutually exclusive"):
            cli.cmd_push(parse(argv))

    def test_env_token_calibration_note(self, push_env, monkeypatch, capsys):
        monkeypatch.setenv("HF_TOKEN", "hf_env")
        assert cli.cmd_push(parse(["push", str(push_env.local), "--repo", "me/model"])) == cli.EXIT_OK
        assert "Using HF_TOKEN from the environment" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "code, expected",
        [
            ("INPUT_AUTH_REQUIRED", cli.EXIT_USER_ERROR),
            ("HUB_PUSH_INVALID_REPO", cli.EXIT_USER_ERROR),
            ("HUB_PUSH_NOT_FOUND", cli.EXIT_USER_ERROR),
            ("HUB_PUSH_NETWORK", cli.EXIT_RUNTIME_ERROR),
        ],
    )
    def test_export_error_code_to_exit_code(self, push_env, capsys, code, expected):
        push_env.exc = ExportError("rejected", suggestion="log in first", code=code)
        assert cli.cmd_push(parse(["push", str(push_env.local), "--repo", "me/model"])) == expected
        captured = capsys.readouterr()
        assert "Push failed: rejected" in captured.err
        assert "Suggestion: log in first" in captured.out

    def test_export_error_without_suggestion(self, push_env, capsys):
        push_env.exc = ExportError("rejected")
        assert cli.cmd_push(parse(["push", str(push_env.local), "--repo", "me/model"])) == cli.EXIT_RUNTIME_ERROR
        assert "Suggestion:" not in capsys.readouterr().out

    def test_other_structured_error(self, push_env, capsys):
        push_env.exc = BackpropagateError("hub down", suggestion="retry later")
        assert cli.cmd_push(parse(["push", str(push_env.local), "--repo", "me/model"])) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "Push failed: hub down" in captured.err and "Suggestion: retry later" in captured.out

    def test_unexpected_error_redacted_and_verbose(self, push_env, capsys):
        push_env.exc = RuntimeError("Authorization: Bearer abcdef1234567890")
        argv = ["push", str(push_env.local), "--repo", "me/model"]
        assert cli.cmd_push(parse(argv)) == cli.EXIT_RUNTIME_ERROR
        captured = capsys.readouterr()
        assert "abcdef1234567890" not in captured.err
        assert "Run with --verbose" in captured.out
        args = parse(argv)
        args.verbose = True
        assert cli.cmd_push(args) == cli.EXIT_RUNTIME_ERROR
        assert "Traceback" in capsys.readouterr().err


class TestHumanizeTimestamp:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, "-"),
            ("", "-"),
            ("2026-05-21T13:42:18.123456", "2026-05-21 13:42"),
            ("2026-05-21T13:42:18", "2026-05-21 13:42"),
            ("short", "short"),
        ],
    )
    def test_humanize(self, value, expected):
        assert cli._humanize_timestamp(value) == expected
