# ui-v2 P3: CLI parity for the web UI's run form.
"""Every flag the Single run / Multi-run form passes must be a real CLI flag
that reaches the Trainer.

Mock boundary: the Trainer / MultiRunTrainer classes (model load + the training
loop are the GPU boundary) and the GPU temperature source. Argument parsing,
validation, the kwarg-introspection filter, MultiRunConfig, the job-event files,
SFTConfig assembly and the callbacks are real.
"""

from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backpropagate import cli
from backpropagate.config import settings
from backpropagate.exceptions import UserInputError
from backpropagate.job_events import EVENTS_FILENAME, JobEventWriter
from tests.helpers.cli_cov_support import last_json, parse

P3_TRAINER_KEYS = {
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "load_in_4bit",
    "run_name",
    "gradient_checkpointing",
}


def _rows(run_dir):
    path = run_dir / EVENTS_FILENAME
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


class _Control:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = False


class _State:
    def __init__(self, step=1):
        self.global_step = step
        self.max_steps = 10
        self.epoch = 0.0


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_trainer(record, *, hot_steps=0, hot_temp=90.0):
    """A Trainer double whose __init__ takes **kwargs (so the introspection
    filter forwards everything) and whose train() can drive the callbacks."""

    class FakeTrainer:
        def __init__(self, **kwargs):
            record["init"] = kwargs

        def train(self, **kwargs):
            record["train"] = kwargs
            for cb in kwargs.get("extra_callbacks", []):
                if hasattr(cb, "poll_s"):
                    cb.poll_s = 0.0  # poll on every step in the test
                    record["temp_cb"] = cb
            for i in range(hot_steps):
                control = _Control()
                for cb in kwargs.get("extra_callbacks", []):
                    cb.on_step_end(argparse.Namespace(), _State(i + 1), control)
                record["control"] = control
            return SimpleNamespace(
                final_loss=0.25, duration_seconds=1.0, run_id=None, steps=hot_steps
            )

        def save(self, out):
            return out

    return FakeTrainer


def _fake_multi(record, *, hot_steps=0):
    class FakeMulti:
        def __init__(self, model, config, on_run_complete, resume_from=None, **extra):
            record["init"] = {"config": config, "extra": extra}
            self.aborted_with = None
            record["trainer"] = self

        def abort(self, reason="x"):
            self.aborted_with = reason

        def run(self, data):
            for cb in record["init"]["extra"].get("extra_callbacks", []):
                if hasattr(cb, "poll_s"):
                    cb.poll_s = 0.0
            for i in range(hot_steps):
                control = _Control()
                for cb in record["init"]["extra"].get("extra_callbacks", []):
                    cb.on_step_end(argparse.Namespace(), _State(i + 1), control)
            return SimpleNamespace(
                total_runs=1,
                final_loss=0.5,
                total_duration_seconds=1.0,
                final_checkpoint_path=None,
                failed_runs=0,
            )

    return FakeMulti


@pytest.fixture
def hot_gpu(monkeypatch):
    monkeypatch.setattr("backpropagate.gpu_safety._read_device_temperature_c", lambda: 90.0)


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------

P3_FLAGS = [
    "--lora-alpha",
    "32",
    "--lora-dropout",
    "0.1",
    "--target-modules",
    "q_proj, v_proj",
    "--no-4bit",
    "--run-name",
    "exp-1",
    "--no-gradient-checkpointing",
    "--gpu-max-temp",
    "85",
]


@pytest.mark.parametrize("sub", ["train", "multi-run"])
def test_every_p3_flag_parses_on_train_and_multi_run(sub):
    args = parse([sub, "--data", "d", *P3_FLAGS])
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.1
    assert args.target_modules == "q_proj, v_proj"
    assert args.no_4bit is True
    assert args.run_name == "exp-1"
    assert args.no_gradient_checkpointing is True
    assert args.gpu_max_temp == 85.0


@pytest.mark.parametrize("sub", ["train", "multi-run"])
def test_p3_flags_default_to_unset(sub):
    args = parse([sub, "--data", "d"])
    assert args.lora_alpha is None and args.lora_dropout is None
    assert args.target_modules is None and args.run_name is None
    assert args.no_4bit is False and args.no_gradient_checkpointing is False
    assert args.gpu_max_temp is None


def test_multi_run_gets_the_basic_training_knobs():
    args = parse(["multi-run", "--data", "d", "--lr", "1e-4", "--batch-size", "4", "--lora-r", "8"])
    assert (args.lr, args.batch_size, args.lora_r) == (1e-4, 4, 8)
    args = parse(["multi-run", "--data", "d", "--batch-size", "auto"])
    assert args.batch_size == "auto"
    args = parse(["multi-run", "--data", "d"])
    assert (args.lr, args.batch_size, args.lora_r) == (None, None, None)


@pytest.mark.parametrize(
    "bad",
    [
        ["--lora-alpha", "0"],
        ["--lora-alpha", "-4"],
        ["--lora-alpha", "x"],
        ["--lora-dropout", "1"],
        ["--lora-dropout", "1.5"],
        ["--lora-dropout", "-0.1"],
        ["--lora-dropout", "nan"],
        ["--gpu-max-temp", "49.9"],
        ["--gpu-max-temp", "105.1"],
        ["--gpu-max-temp", "hot"],
    ],
)
@pytest.mark.parametrize("sub", ["train", "multi-run"])
def test_numeric_flag_validation_rejects_bad_values(sub, bad, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.create_parser().parse_args([sub, "--data", "d", *bad])
    assert exc.value.code == 2
    assert bad[0] in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value", [("--lr", "0"), ("--lora-r", "0"), ("--batch-size", "0"), ("--batch-size", "foo")]
)
def test_multi_run_numeric_flags_reuse_the_train_validators(flag, value):
    with pytest.raises(SystemExit) as exc:
        cli.create_parser().parse_args(["multi-run", "--data", "d", flag, value])
    assert exc.value.code == 2


def test_boundary_values_are_accepted():
    args = parse(["train", "--data", "d", "--lora-dropout", "0", "--gpu-max-temp", "50"])
    assert args.lora_dropout == 0.0 and args.gpu_max_temp == 50.0
    args = parse(["train", "--data", "d", "--lora-dropout", "0.999", "--gpu-max-temp", "105"])
    assert args.gpu_max_temp == 105.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("all-linear", "all-linear"),
        (" ALL-LINEAR ", "all-linear"),
        ("q_proj", ["q_proj"]),
        ("q_proj,v_proj", ["q_proj", "v_proj"]),
        (" q_proj , v_proj ", ["q_proj", "v_proj"]),
        ("q_proj,q_proj,v_proj", ["q_proj", "v_proj"]),
        ("model.layers.0.self_attn.q_proj", ["model.layers.0.self_attn.q_proj"]),
    ],
)
def test_target_modules_parsing(raw, expected):
    assert cli._parse_target_modules(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "q_proj,,v_proj",
        "q_proj,",
        ",q_proj",
        "q proj",
        "q-proj",
        "1abc",
        "a..b",
        "q_proj;rm",
        "all-linear,q_proj",
    ],
)
def test_target_modules_rejects_malformed_values_with_the_structured_input_error(raw):
    with pytest.raises(UserInputError) as exc:
        cli._parse_target_modules(raw)
    assert exc.value.code == "INPUT_VALIDATION_FAILED"
    assert exc.value.suggestion


def test_bad_target_modules_is_a_user_error_with_the_code_printed(capsys):
    args = parse(["train", "--data", "d", "--target-modules", "q_proj,,v_proj"])
    assert cli.cmd_train(args) == cli.EXIT_USER_ERROR
    captured = capsys.readouterr()
    assert "INPUT_VALIDATION_FAILED" in captured.out + captured.err


def test_bad_target_modules_on_multi_run_is_a_user_error(capsys):
    args = parse(["multi-run", "--data", "d", "--target-modules", "bad name"])
    assert cli.cmd_multi_run(args) == cli.EXIT_USER_ERROR
    captured = capsys.readouterr()
    assert "INPUT_VALIDATION_FAILED" in captured.out + captured.err


# ---------------------------------------------------------------------------
# train: flags reach the Trainer constructor
# ---------------------------------------------------------------------------


def test_train_forwards_every_p3_flag_to_the_trainer(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record))
    args = parse(["train", "--data", "d", *P3_FLAGS])
    assert cli.cmd_train(args) == cli.EXIT_OK
    init = record["init"]
    assert init["lora_alpha"] == 32
    assert init["lora_dropout"] == 0.1
    assert init["target_modules"] == ["q_proj", "v_proj"]
    assert init["load_in_4bit"] is False
    assert init["run_name"] == "exp-1"
    assert init["gradient_checkpointing"] is False
    # --lr / --batch-size keep their existing path; no --lora-r means the
    # trainer resolves the rank (settings, or the preset that fits the GPU).
    assert init["lora_r"] is None and init["learning_rate"] == 2e-4 and init["batch_size"] == "auto"


def test_train_all_linear_is_forwarded_as_the_literal(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record))
    assert (
        cli.cmd_train(parse(["train", "--data", "d", "--target-modules", "all-linear"]))
        == cli.EXIT_OK
    )
    assert record["init"]["target_modules"] == "all-linear"


def test_train_defaults_forward_none_of_the_new_kwargs(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record))
    assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_OK
    assert not (P3_TRAINER_KEYS & set(record["init"]))
    assert "extra_callbacks" not in record["train"]  # no UI, no temp limit


def test_train_with_a_real_signature_drops_unsupported_new_kwargs(monkeypatch):
    """A Trainer without the new params (older build) must not crash on them."""
    seen: dict = {}

    class OldTrainer:
        def __init__(self, model, lora_r, learning_rate, batch_size, output_dir, use_unsloth):
            seen["ok"] = True

        def train(self, dataset, steps, samples, callback, resume_from):
            return SimpleNamespace(final_loss=0.1, duration_seconds=1.0, run_id=None)

        def save(self, out):
            return out

    monkeypatch.setattr("backpropagate.trainer.Trainer", OldTrainer)
    flags = [f for f in P3_FLAGS if f not in ("--gpu-max-temp", "85")]
    assert cli.cmd_train(parse(["train", "--data", "d", *flags])) == cli.EXIT_OK
    assert seen["ok"]


def test_no_4bit_in_full_mode_is_not_an_error(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record))
    assert (
        cli.cmd_train(parse(["train", "--data", "d", "--mode", "full", "--no-4bit"])) == cli.EXIT_OK
    )


# ---------------------------------------------------------------------------
# multi-run: flags reach MultiRunConfig / the inner Trainer
# ---------------------------------------------------------------------------


def test_multi_run_forwards_every_flag_into_the_config(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _fake_multi(record))
    args = parse(
        [
            "multi-run",
            "--data",
            "d",
            "--lr",
            "1e-4",
            "--batch-size",
            "4",
            "--lora-r",
            "8",
            *P3_FLAGS,
        ]
    )
    assert cli.cmd_multi_run(args) == cli.EXIT_OK
    cfg = record["init"]["config"]
    assert cfg.lora_r == 8
    assert cfg.batch_size == 4
    assert cfg.initial_lr == 1e-4
    assert cfg.final_lr == pytest.approx(2.5e-5)
    assert cfg.lora_alpha == 32 and cfg.lora_dropout == 0.1
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.load_in_4bit is False
    assert cfg.run_name == "exp-1"
    assert cfg.gradient_checkpointing is False
    # none of them leaked to MultiRunTrainer kwargs (config owns them)
    assert not (P3_TRAINER_KEYS & set(record["init"]["extra"]))


def test_multi_run_defaults_leave_the_config_untouched(monkeypatch):
    from backpropagate.multi_run import MultiRunConfig

    record: dict = {}
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _fake_multi(record))
    assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_OK
    cfg = record["init"]["config"]
    for name in (
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "batch_size",
        "target_modules",
        "load_in_4bit",
        "gradient_checkpointing",
        "run_name",
    ):
        assert getattr(cfg, name) is None, name
    assert (cfg.initial_lr, cfg.final_lr) == (
        MultiRunConfig().initial_lr,
        MultiRunConfig().final_lr,
    )
    assert "extra_callbacks" not in record["init"]["extra"]


def test_multi_run_batch_size_auto_is_the_default_not_a_forced_value(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _fake_multi(record))
    cli.cmd_multi_run(parse(["multi-run", "--data", "d", "--batch-size", "auto"]))
    assert record["init"]["config"].batch_size is None


def test_lr_decay_ratio_matches_the_multi_run_config_defaults():
    from backpropagate.multi_run import MultiRunConfig

    cfg = MultiRunConfig()
    assert cfg.final_lr / cfg.initial_lr == pytest.approx(cli._MULTI_RUN_FINAL_LR_RATIO)


def test_multi_run_config_overrides_reach_the_inner_trainer_kwargs(tmp_path):
    from tests.test_multi_run_cov_support import make_trainer

    mrt = make_trainer(
        tmp_path,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        batch_size=2,
        target_modules=["q_proj"],
        load_in_4bit=False,
        gradient_checkpointing=False,
    )
    assert mrt._trainer_override_kwargs() == {
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj"],
        "load_in_4bit": False,
        "gradient_checkpointing": False,
        "batch_size": 2,
    }
    assert make_trainer(tmp_path)._trainer_override_kwargs() == {}
    assert make_trainer(tmp_path, batch_size="auto")._trainer_override_kwargs() == {}


def test_multi_run_run_name_suffix_and_checkpointing_reach_each_run(tmp_path, monkeypatch):
    from backpropagate.multi_run import MergeMode
    from tests.test_multi_run_cov_support import (
        FakeInnerTrainer,
        install_fake_sft,
        make_fake_sft,
        make_trainer,
        text_dataset,
    )

    fake = make_fake_sft()
    install_fake_sft(monkeypatch, fake)
    inner = FakeInnerTrainer(report_to="wandb")
    inner._gradient_checkpointing_override = False
    mrt = make_trainer(tmp_path, inner=inner, merge_mode=MergeMode.SIMPLE, run_name="exp")
    mrt._execute_run(3, text_dataset(40), tmp_path)
    assert fake.created[0].args.run_name == "exp-run3"
    assert fake.created[0].args.gradient_checkpointing is False


def test_multi_run_run_name_default_is_unchanged(tmp_path, monkeypatch):
    from backpropagate.multi_run import MergeMode
    from tests.test_multi_run_cov_support import (
        FakeInnerTrainer,
        install_fake_sft,
        make_fake_sft,
        make_trainer,
        text_dataset,
    )

    fake = make_fake_sft()
    install_fake_sft(monkeypatch, fake)
    mrt = make_trainer(
        tmp_path, inner=FakeInnerTrainer(report_to="wandb"), merge_mode=MergeMode.SIMPLE
    )
    mrt._execute_run(3, text_dataset(40), tmp_path)
    assert fake.created[0].args.run_name == "backprop-abcdef012345-run-003"


# ---------------------------------------------------------------------------
# Trainer: the new kwargs take effect in BOTH loaders + the SFT config
# ---------------------------------------------------------------------------


def _tiny_trainer(**kw):
    from backpropagate.trainer import Trainer

    kw.setdefault("use_unsloth", False)
    kw.setdefault("batch_size", 2)
    kw.setdefault("report_to", "none")
    kw.setdefault("model", "acme/Tiny-1B")
    return Trainer(**kw)


def _run_transformers_loader(trainer):
    model = MagicMock()
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as from_pre,
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tok),
        patch("transformers.BitsAndBytesConfig") as bnb,
        patch("peft.prepare_model_for_kbit_training", return_value=model) as prep,
        patch("peft.get_peft_model", return_value=MagicMock()),
        patch("peft.LoraConfig") as lora_config,
    ):
        trainer._load_with_transformers()
    return SimpleNamespace(from_pre=from_pre, bnb=bnb, prep=prep, lora_config=lora_config)


def _tiny_linear_model():
    """A real module with one Linear + an lm_head so "all-linear" resolves."""
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 8)

        def get_output_embeddings(self):
            return self.lm_head

    return M()


def _run_unsloth_loader(trainer, model=None):
    from backpropagate import feature_flags

    model = model if model is not None else _tiny_linear_model()
    fast = MagicMock()
    fast.from_pretrained.return_value = (model, MagicMock())
    fast.get_peft_model.return_value = model
    with (
        patch.dict("sys.modules", {"unsloth": MagicMock(FastLanguageModel=fast)}),
        patch.dict(feature_flags.FEATURES, {"unsloth": True}),
        patch("unsloth.FastLanguageModel", fast),
    ):
        trainer._load_with_unsloth()
    return fast


class TestTargetModulesReachBothLoaders:
    def test_transformers_loader_uses_the_override(self):
        run = _run_transformers_loader(_tiny_trainer(target_modules=["q_proj", "v_proj"]))
        assert run.lora_config.call_args.kwargs["target_modules"] == ["q_proj", "v_proj"]

    def test_transformers_loader_default_is_the_settings_value(self):
        run = _run_transformers_loader(_tiny_trainer())
        assert run.lora_config.call_args.kwargs["target_modules"] == settings.lora.target_modules

    def test_override_wins_over_a_changed_settings_value(self):
        trainer = _tiny_trainer(target_modules="all-linear")
        with patch("backpropagate.trainer.settings.lora.target_modules", ["o_proj"]):
            run = _run_transformers_loader(trainer)
        assert run.lora_config.call_args.kwargs["target_modules"] == "all-linear"

    def test_unsloth_loader_uses_the_override(self):
        fast = _run_unsloth_loader(
            _tiny_trainer(use_unsloth=True, target_modules=["q_proj", "v_proj"])
        )
        assert fast.get_peft_model.call_args.kwargs["target_modules"] == ["q_proj", "v_proj"]

    def test_unsloth_loader_expands_all_linear_from_the_model(self):
        fast = _run_unsloth_loader(_tiny_trainer(use_unsloth=True, target_modules="all-linear"))
        assert fast.get_peft_model.call_args.kwargs["target_modules"] == ["proj"]

    def test_unsloth_loader_default_is_the_settings_value(self):
        fast = _run_unsloth_loader(_tiny_trainer(use_unsloth=True))
        assert settings.lora.target_modules == "all-linear"
        assert fast.get_peft_model.call_args.kwargs["target_modules"] == ["proj"]


class TestNoGradientCheckpointingReachesBothLoadersAndTheConfig:
    def test_transformers_loader_turns_checkpointing_off(self):
        run = _run_transformers_loader(_tiny_trainer(gradient_checkpointing=False))
        assert run.prep.call_args.kwargs == {"use_gradient_checkpointing": False}

    def test_transformers_loader_default_call_is_unchanged(self):
        run = _run_transformers_loader(_tiny_trainer())
        assert run.prep.call_args.kwargs == {}

    def test_gradient_checkpointing_true_keeps_the_default_call(self):
        run = _run_transformers_loader(_tiny_trainer(gradient_checkpointing=True))
        assert run.prep.call_args.kwargs == {}

    def test_unsloth_loader_turns_checkpointing_off(self):
        fast = _run_unsloth_loader(_tiny_trainer(use_unsloth=True, gradient_checkpointing=False))
        assert fast.get_peft_model.call_args.kwargs["use_gradient_checkpointing"] is False

    def test_unsloth_loader_default_is_the_settings_value(self):
        fast = _run_unsloth_loader(_tiny_trainer(use_unsloth=True))
        assert (
            fast.get_peft_model.call_args.kwargs["use_gradient_checkpointing"]
            == settings.lora.use_gradient_checkpointing
        )

    def test_sft_config_sets_gradient_checkpointing_false(self, tmp_path):
        cfg = _tiny_trainer(
            gradient_checkpointing=False, output_dir=str(tmp_path)
        )._build_training_args(steps=2, report_to="none", run_name=None)
        assert cfg.gradient_checkpointing is False

    def test_sft_config_default_is_what_it_was(self, tmp_path):
        default = _tiny_trainer(output_dir=str(tmp_path))._build_training_args(
            steps=2, report_to="none", run_name=None
        )
        explicit_true = _tiny_trainer(
            gradient_checkpointing=True, output_dir=str(tmp_path)
        )._build_training_args(steps=2, report_to="none", run_name=None)
        assert default.gradient_checkpointing == explicit_true.gradient_checkpointing

    def test_full_mode_keeps_checkpointing_and_warns_once(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="backpropagate.trainer"):
            trainer = _tiny_trainer(
                mode="full", gradient_checkpointing=False, output_dir=str(tmp_path)
            )
        warnings_ = [
            r for r in caplog.records if "gradient_checkpointing=False is ignored" in r.getMessage()
        ]
        assert len(warnings_) == 1
        cfg = trainer._build_training_args(steps=2, report_to="none", run_name=None)
        assert cfg.gradient_checkpointing is True
        assert trainer._gradient_checkpointing_disabled() is False


class TestNo4bitReachesBothLoaders:
    def test_transformers_loader_skips_bitsandbytes(self):
        run = _run_transformers_loader(_tiny_trainer(load_in_4bit=False))
        run.bnb.assert_not_called()
        kwargs = run.from_pre.call_args.kwargs
        assert "quantization_config" not in kwargs and "dtype" in kwargs

    def test_transformers_loader_default_is_4bit(self):
        run = _run_transformers_loader(_tiny_trainer())
        run.bnb.assert_called_once()
        assert run.bnb.call_args.kwargs["load_in_4bit"] is True

    def test_unsloth_loader_passes_load_in_4bit_false(self):
        fast = _run_unsloth_loader(_tiny_trainer(use_unsloth=True, load_in_4bit=False))
        assert fast.from_pretrained.call_args.kwargs["load_in_4bit"] is False

    def test_unsloth_loader_default_is_4bit(self):
        fast = _run_unsloth_loader(_tiny_trainer(use_unsloth=True))
        assert fast.from_pretrained.call_args.kwargs["load_in_4bit"] is True

    def test_full_mode_ignores_it_without_error(self):
        trainer = _tiny_trainer(mode="full", load_in_4bit=False)
        assert trainer.mode == "full"


def test_run_name_is_stored_on_the_trainer():
    assert _tiny_trainer(run_name="exp").run_name == "exp"
    assert _tiny_trainer().run_name is None


# ---------------------------------------------------------------------------
# GpuTempStopCallback
# ---------------------------------------------------------------------------


def _temp_cb(temps, **kw):
    """A callback fed a scripted temperature sequence (None = no reading)."""
    from backpropagate.gpu_safety import build_gpu_temp_stop_callback

    feed = iter(temps)
    trips: list[str] = []
    kw.setdefault("poll_s", 0.0)
    cb = build_gpu_temp_stop_callback(
        85.0, on_trip=trips.append, read_temp_c=lambda: next(feed), **kw
    )
    return cb, trips


def _step(cb, control=None, step=1):
    control = control or _Control()
    cb.on_step_end(argparse.Namespace(), _State(step), control)
    return control


class TestGpuTempStopCallback:
    def test_trips_after_consecutive_hot_polls_and_stops_cooperatively(self):
        cb, trips = _temp_cb([86, 87, 88])
        c1, c2, c3 = _step(cb), _step(cb), _step(cb)
        assert not (c1.should_training_stop or c2.should_training_stop)
        assert c3.should_training_stop is True and c3.should_save is True
        assert trips == ["GPU at 88 °C, above the 85 °C limit"]
        assert cb.tripped and cb.trip_reason == trips[0]

    def test_a_single_spike_does_not_trip(self):
        cb, trips = _temp_cb([95, 70, 95, 70, 95, 70])
        controls = [_step(cb) for _ in range(6)]
        assert not any(c.should_training_stop for c in controls)
        assert trips == []

    def test_a_cool_reading_resets_the_streak(self):
        cb, trips = _temp_cb([90, 90, 60, 90, 90])
        controls = [_step(cb) for _ in range(5)]
        assert not any(c.should_training_stop for c in controls) and trips == []

    def test_the_limit_itself_counts_as_hot(self):
        cb, trips = _temp_cb([85, 85, 85])
        _step(cb), _step(cb)
        assert _step(cb).should_training_stop is True
        assert trips

    def test_no_readings_never_trips(self):
        cb, trips = _temp_cb([None] * 10)
        controls = [_step(cb) for _ in range(10)]
        assert not any(c.should_training_stop or c.should_save for c in controls)
        assert trips == [] and not cb.tripped

    def test_missing_readings_do_not_count_as_hot(self):
        cb, trips = _temp_cb([90, None, 90, None, 90])
        controls = [_step(cb) for _ in range(5)]
        # 3 hot readings arrive (gaps skipped), so it trips on the 5th poll
        assert controls[-1].should_training_stop is True and len(trips) == 1

    def test_on_trip_is_called_exactly_once(self):
        cb, trips = _temp_cb([90] * 8)
        controls = [_step(cb) for _ in range(8)]
        assert len(trips) == 1
        # the stop request is re-asserted on later steps
        assert controls[-1].should_training_stop and controls[-1].should_save

    def test_poll_interval_gates_the_sampling(self):
        from backpropagate.gpu_safety import build_gpu_temp_stop_callback

        now = [0.0]
        reads: list[int] = []

        def read():
            reads.append(1)
            return 90.0

        cb = build_gpu_temp_stop_callback(
            85.0, poll_s=5.0, consecutive=3, read_temp_c=read, clock=lambda: now[0]
        )
        for t in (0.0, 1.0, 2.0, 4.9):
            now[0] = t
            _step(cb)
        assert len(reads) == 1  # one poll in the first 5 s
        for t in (5.0, 10.0):
            now[0] = t
            control = _step(cb)
        assert len(reads) == 3 and control.should_training_stop is True

    def test_new_run_starts_a_clean_streak(self):
        cb, trips = _temp_cb([90, 90, 90, 90])
        _step(cb), _step(cb)
        cb.on_train_begin(argparse.Namespace(), _State(0), _Control())
        _step(cb)
        assert trips == []  # 2 + reset + 1 hot readings never reached 3 in a row

    def test_on_trip_failure_does_not_break_the_stop(self):
        from backpropagate.gpu_safety import build_gpu_temp_stop_callback

        def boom(reason):
            raise RuntimeError("hook down")

        feed = iter([90, 90, 90])
        cb = build_gpu_temp_stop_callback(
            85.0, on_trip=boom, poll_s=0.0, read_temp_c=lambda: next(feed)
        )
        _step(cb), _step(cb)
        assert _step(cb).should_training_stop is True

    def test_logs_a_structured_warning(self, caplog):
        cb, _ = _temp_cb([90, 90, 90])
        with caplog.at_level(logging.WARNING, logger="backpropagate.gpu_safety"):
            _step(cb), _step(cb), _step(cb)
        rec = [r for r in caplog.records if getattr(r, "event", "") == "gpu_temp_stop"]
        assert len(rec) == 1 and rec[0].temp_c == 90 and rec[0].max_temp_c == 85.0

    def test_default_source_reads_get_system_gpu_readings(self, monkeypatch):
        from backpropagate import gpu_safety

        monkeypatch.setattr(
            gpu_safety,
            "get_system_gpu_readings",
            lambda *a, **k: gpu_safety.SystemGpuReadings(temperature_c=91.0),
        )
        assert gpu_safety._read_device_temperature_c() == 91.0
        monkeypatch.setattr(gpu_safety, "get_system_gpu_readings", lambda *a, **k: None)
        assert gpu_safety._read_device_temperature_c() is None

        def boom(*a, **k):
            raise OSError("nvml")

        monkeypatch.setattr(gpu_safety, "get_system_gpu_readings", boom)
        assert gpu_safety._read_device_temperature_c() is None

    def test_class_is_importable_lazily_and_is_an_hf_callback(self):
        from transformers import TrainerCallback

        from backpropagate.gpu_safety import GpuTempStopCallback

        assert issubclass(GpuTempStopCallback, TrainerCallback)
        with pytest.raises(ImportError):
            from backpropagate.gpu_safety import NoSuchThing  # noqa: F401


# ---------------------------------------------------------------------------
# job events: safety + terminal event
# ---------------------------------------------------------------------------


def test_job_event_writer_safety_event(tmp_path):
    writer = JobEventWriter(tmp_path)
    writer.safety("GPU at 87 °C, above the 85 °C limit")
    row = _rows(tmp_path)[-1]
    assert row["kind"] == "safety"
    assert row["reason"] == "GPU at 87 °C, above the 85 °C limit"
    assert "ts" in row


def test_done_event_carries_reason_only_when_given(tmp_path):
    writer = JobEventWriter(tmp_path)
    writer.done(status="stopped", steps_done=3, output_path="/o", reason="too hot")
    writer.done(status="done", steps_done=3)
    rows = _rows(tmp_path)
    assert rows[0]["reason"] == "too hot" and rows[0]["status"] == "stopped"
    assert "reason" not in rows[1]


def test_run_as_ui_job_reports_stopped_with_the_safety_reason(tmp_path):
    def body(args):
        args._ui_job.safety_reason = "GPU at 90 °C, above the 85 °C limit"
        return cli.EXIT_OK

    assert (
        cli._run_as_ui_job(argparse.Namespace(ui_run_dir=str(tmp_path)), body, "loading")
        == cli.EXIT_OK
    )
    done = [r for r in _rows(tmp_path) if r["kind"] == "done"][-1]
    assert done["status"] == "stopped"
    assert done["reason"] == "GPU at 90 °C, above the 85 °C limit"
    assert json.loads((tmp_path / "job.json").read_text())["status"] == "stopped"


def test_run_as_ui_job_without_a_trip_is_unchanged(tmp_path):
    cli._run_as_ui_job(
        argparse.Namespace(ui_run_dir=str(tmp_path)), lambda a: cli.EXIT_OK, "loading"
    )
    done = [r for r in _rows(tmp_path) if r["kind"] == "done"][-1]
    assert done["status"] == "done" and "reason" not in done


# ---------------------------------------------------------------------------
# --gpu-max-temp end to end through the CLI
# ---------------------------------------------------------------------------

REASON = "GPU at 90 °C, above the 80 °C limit"


def test_train_temp_trip_stops_cooperatively_without_a_ui(monkeypatch, hot_gpu, capsys):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record, hot_steps=3))
    assert cli.cmd_train(parse(["train", "--data", "d", "--gpu-max-temp", "80"])) == cli.EXIT_OK
    assert record["control"].should_training_stop and record["control"].should_save
    assert REASON in capsys.readouterr().out


def test_train_below_the_limit_does_not_stop(monkeypatch, capsys):
    monkeypatch.setattr("backpropagate.gpu_safety._read_device_temperature_c", lambda: 70.0)
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record, hot_steps=5))
    assert cli.cmd_train(parse(["train", "--data", "d", "--gpu-max-temp", "80"])) == cli.EXIT_OK
    assert not record["control"].should_training_stop


def test_train_ui_job_writes_safety_then_stopped_done_with_reason(monkeypatch, hot_gpu, tmp_path):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record, hot_steps=3))
    args = parse(["train", "--data", "d", "--gpu-max-temp", "80", "--ui-run-dir", str(tmp_path)])
    assert cli.cmd_train(args) == cli.EXIT_OK
    rows = _rows(tmp_path)
    safety = [r for r in rows if r["kind"] == "safety"]
    assert len(safety) == 1 and safety[0]["reason"] == REASON
    done = [r for r in rows if r["kind"] == "done"]
    assert len(done) == 1
    assert done[0]["status"] == "stopped" and done[0]["reason"] == REASON
    assert rows.index(safety[0]) < rows.index(done[0])
    assert json.loads((tmp_path / "job.json").read_text())["stop_reason"] == REASON


def test_train_ui_job_without_a_trip_ends_done(monkeypatch, tmp_path):
    record: dict = {}
    monkeypatch.setattr("backpropagate.trainer.Trainer", _fake_trainer(record))
    args = parse(["train", "--data", "d", "--ui-run-dir", str(tmp_path)])
    assert cli.cmd_train(args) == cli.EXIT_OK
    done = [r for r in _rows(tmp_path) if r["kind"] == "done"][-1]
    assert done["status"] == "done" and "reason" not in done
    assert not [r for r in _rows(tmp_path) if r["kind"] == "safety"]


def test_multi_run_temp_trip_aborts_the_whole_session(monkeypatch, hot_gpu, capsys):
    record: dict = {}
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _fake_multi(record, hot_steps=3))
    assert (
        cli.cmd_multi_run(parse(["multi-run", "--data", "d", "--gpu-max-temp", "80"]))
        == cli.EXIT_OK
    )
    assert record["trainer"].aborted_with == REASON
    assert REASON in capsys.readouterr().out


def test_multi_run_ui_job_reports_stopped_with_reason(monkeypatch, hot_gpu, tmp_path):
    record: dict = {}
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _fake_multi(record, hot_steps=3))
    args = parse(
        ["multi-run", "--data", "d", "--gpu-max-temp", "80", "--ui-run-dir", str(tmp_path)]
    )
    assert cli.cmd_multi_run(args) == cli.EXIT_OK
    assert record["trainer"].aborted_with == REASON
    rows = _rows(tmp_path)
    assert [r["reason"] for r in rows if r["kind"] == "safety"] == [REASON]
    done = [r for r in rows if r["kind"] == "done"][-1]
    assert done["status"] == "stopped" and done["reason"] == REASON
    # the UI progress callback and the temperature callback both ride on every run
    cbs = record["init"]["extra"]["extra_callbacks"]
    assert len(cbs) == 2


def test_multi_run_without_the_flag_installs_no_callback(monkeypatch):
    record: dict = {}
    monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", _fake_multi(record))
    cli.cmd_multi_run(parse(["multi-run", "--data", "d"]))
    assert "extra_callbacks" not in record["init"]["extra"]


# ---------------------------------------------------------------------------
# estimate-vram --no-4bit
# ---------------------------------------------------------------------------


def _estimate(argv, capsys):
    assert (
        cli.cmd_estimate_vram(parse(["estimate-vram", "--vram-gb", "24", "--json", *argv]))
        == cli.EXIT_OK
    )
    return last_json(capsys.readouterr().out)


def test_estimate_vram_default_payload_says_quantized(capsys):
    payload = _estimate([], capsys)
    assert payload["quantize_base"] is True
    assert payload["per_config_estimate"] is None  # unchanged: no per-config view without a trigger


def test_estimate_vram_no_4bit_json_and_bigger_base(capsys):
    quant = _estimate(["--batch-size", "1"], capsys)
    full = _estimate(["--batch-size", "1", "--no-4bit"], capsys)
    assert quant["quantize_base"] is True
    assert full["quantize_base"] is False
    # 16-bit weights are well over the 4-bit base's (measured: about 2x, not
    # 4x, since embeddings stay 16-bit and the rest costs ~0.7 bytes/param).
    assert (
        full["per_config_estimate"]["model_weights_gb"]
        > quant["per_config_estimate"]["model_weights_gb"] * 1.8
    )
    assert full["per_config_estimate"]["total_gb"] > quant["per_config_estimate"]["total_gb"]


def test_estimate_vram_no_4bit_alone_triggers_the_per_config_estimate(capsys):
    payload = _estimate(["--no-4bit"], capsys)
    assert payload["quantize_base"] is False
    assert payload["per_config_estimate"] is not None


def test_estimate_vram_still_honours_mode_lora_r_and_batch_size(capsys):
    payload = _estimate(["--mode", "full", "--lora-r", "8", "--batch-size", "2"], capsys)
    assert payload["mode"] == "full"
    assert payload["per_config_estimate"]["mode"] == "full"


def test_estimate_vram_forwards_quantize_base_false_only_when_set(monkeypatch, capsys):
    seen: list[dict] = []

    def fake(**kwargs):
        seen.append(kwargs)
        return {"total_gb": 1.0}

    monkeypatch.setattr("backpropagate.trainer.estimate_vram", fake)
    _estimate(["--batch-size", "2"], capsys)
    _estimate(["--batch-size", "2", "--no-4bit"], capsys)
    assert "quantize_base" not in seen[0]
    assert seen[1]["quantize_base"] is False
