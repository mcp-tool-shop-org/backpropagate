"""report_to="auto" must not wire W&B in when it has no credentials.

Bug (2026-10-01): ``backprop train`` on a ``[monitoring]`` / ``[full]`` install
crashed at the first step with wandb's ``UsageError: No API key configured``,
because "auto" added "wandb" whenever the package was importable. These tests
pin the fix: auto skips an unconfigured W&B with one INFO line, an explicit W&B
request without credentials fails fast with CONFIG_INVALID_SETTING, and
``--report-to`` reaches the Trainer from ``train`` and ``multi-run``.

No GPU, no wandb import, no network: credentials are faked through the same
env vars and netrc file wandb itself reads.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backpropagate import cli, feature_flags
from backpropagate import trainer as trainer_mod
from backpropagate.exceptions import ConfigurationError
from tests.helpers.cli_cov_support import parse

_FAKE_KEY = "x" * 40


@pytest.fixture(autouse=True)
def no_wandb_credentials(monkeypatch, tmp_path):
    """Start every test with no W&B credentials anywhere wandb would look."""
    for var in (
        "WANDB_API_KEY",
        "WANDB_IDENTITY_TOKEN_FILE",
        "WANDB_MODE",
        "WANDB_BASE_URL",
        "NETRC",
    ):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    # Path.expanduser reads USERPROFILE on Windows and HOME elsewhere.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _features(**flags):
    return patch.dict(feature_flags.FEATURES, flags, clear=False)


def _trainer(**kw):
    with patch("torch.cuda.is_available", return_value=False):
        return trainer_mod.Trainer(**kw)


# ---------------------------------------------------------------------------
# _wandb_credential_source
# ---------------------------------------------------------------------------


class TestWandbCredentialSource:
    def test_nothing_configured(self):
        assert trainer_mod._wandb_credential_source() is None

    def test_api_key_env(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", _FAKE_KEY)
        assert trainer_mod._wandb_credential_source() == "WANDB_API_KEY"

    def test_blank_api_key_is_not_a_credential(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "   ")
        assert trainer_mod._wandb_credential_source() is None

    def test_identity_token_file_env(self, monkeypatch):
        monkeypatch.setenv("WANDB_IDENTITY_TOKEN_FILE", "/tmp/token")
        assert trainer_mod._wandb_credential_source() == "WANDB_IDENTITY_TOKEN_FILE"

    @pytest.mark.parametrize("mode", ["offline", "disabled", "dryrun", "OFFLINE"])
    def test_keyless_modes(self, monkeypatch, mode):
        monkeypatch.setenv("WANDB_MODE", mode)
        assert trainer_mod._wandb_credential_source() == f"WANDB_MODE={mode.lower()}"

    def test_online_mode_alone_is_not_enough(self, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "online")
        assert trainer_mod._wandb_credential_source() is None

    @pytest.mark.parametrize("name", [".netrc", "_netrc"])
    def test_netrc_from_wandb_login(self, no_wandb_credentials, name):
        netrc_path = no_wandb_credentials / name
        netrc_path.write_text(f"machine api.wandb.ai\n  login user\n  password {_FAKE_KEY}\n")
        assert trainer_mod._wandb_credential_source() == str(netrc_path)

    def test_netrc_for_another_host_does_not_count(self, no_wandb_credentials):
        (no_wandb_credentials / ".netrc").write_text(
            "machine github.com\n  login user\n  password secret\n"
        )
        assert trainer_mod._wandb_credential_source() is None

    def test_netrc_env_and_custom_base_url(self, monkeypatch, tmp_path):
        netrc_path = tmp_path / "custom-netrc"
        netrc_path.write_text(f"machine wandb.example.com\n  login user\n  password {_FAKE_KEY}\n")
        monkeypatch.setenv("NETRC", str(netrc_path))
        monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example.com")
        assert trainer_mod._wandb_credential_source() == str(netrc_path)

    def test_unparseable_netrc_is_ignored(self, no_wandb_credentials):
        (no_wandb_credentials / ".netrc").write_text("machine\n")
        assert trainer_mod._wandb_credential_source() is None


# ---------------------------------------------------------------------------
# Trainer._resolve_report_to
# ---------------------------------------------------------------------------


class TestAutoSkipsUnconfiguredWandb:
    def test_auto_wandb_importable_but_unconfigured_is_skipped(self, caplog):
        trainer = _trainer()
        with _features(wandb=True, tensorboard=False, mlflow=False), caplog.at_level(
            logging.INFO, logger="backpropagate.trainer"
        ):
            assert trainer._resolve_report_to() == "none"
        skip_lines = [r for r in caplog.records if "wandb is installed but not logged in" in r.getMessage()]
        assert len(skip_lines) == 1
        assert skip_lines[0].levelno == logging.INFO
        assert "wandb login" in skip_lines[0].getMessage()

    def test_skip_line_logged_once_per_trainer(self, caplog):
        trainer = _trainer()
        with _features(wandb=True, tensorboard=False, mlflow=False), caplog.at_level(
            logging.INFO, logger="backpropagate.trainer"
        ):
            trainer._resolve_report_to()
            trainer._resolve_report_to()  # multi-run resolves once per run
        assert sum("not logged in" in r.getMessage() for r in caplog.records) == 1

    def test_auto_keeps_other_trackers_when_wandb_skipped(self):
        trainer = _trainer()
        with _features(wandb=True, tensorboard=True, mlflow=True):
            assert trainer._resolve_report_to() == ["tensorboard", "mlflow"]

    def test_auto_with_api_key_includes_wandb(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", _FAKE_KEY)
        trainer = _trainer()
        with _features(wandb=True, tensorboard=False, mlflow=False):
            assert trainer._resolve_report_to() == ["wandb"]

    def test_auto_with_offline_mode_includes_wandb(self, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "offline")
        trainer = _trainer()
        with _features(wandb=True, tensorboard=False, mlflow=False):
            assert trainer._resolve_report_to() == ["wandb"]

    def test_auto_with_netrc_login_includes_wandb(self, no_wandb_credentials):
        (no_wandb_credentials / ".netrc").write_text(
            f"machine api.wandb.ai\n  login user\n  password {_FAKE_KEY}\n"
        )
        trainer = _trainer()
        with _features(wandb=True, tensorboard=False, mlflow=False):
            assert trainer._resolve_report_to() == ["wandb"]


class TestExplicitWandb:
    def test_explicit_wandb_honored_when_configured(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", _FAKE_KEY)
        trainer = _trainer(report_to="wandb")
        with _features(wandb=True):
            assert trainer._resolve_report_to() == ["wandb"]

    @pytest.mark.parametrize("intent", ["wandb", "WandB", ["wandb", "tensorboard"]])
    def test_explicit_wandb_unconfigured_errors_clearly(self, intent):
        trainer = _trainer(report_to=intent)
        with _features(wandb=True), pytest.raises(ConfigurationError) as excinfo:
            trainer._resolve_report_to()
        err = excinfo.value
        assert err.code == "CONFIG_INVALID_SETTING"
        assert "wandb" in err.message and "API key" in err.message
        assert "wandb login" in err.suggestion
        assert "--report-to none" in err.suggestion

    def test_explicit_wandb_not_installed_keeps_existing_contract(self):
        # Without the package the credential check is skipped; TRL raises its
        # own ImportError later, as before this fix.
        trainer = _trainer(report_to="wandb")
        with _features(wandb=False):
            assert trainer._resolve_report_to() == ["wandb"]

    def test_explicit_tensorboard_needs_no_credentials(self):
        trainer = _trainer(report_to="tensorboard")
        with _features(wandb=True, tensorboard=True):
            assert trainer._resolve_report_to() == ["tensorboard"]

    def test_train_fails_before_model_load(self):
        trainer = _trainer(report_to="wandb", use_unsloth=False)
        with _features(wandb=True), patch.object(
            trainer_mod.Trainer, "load_model", side_effect=AssertionError("model loaded")
        ) as load_model, pytest.raises(ConfigurationError):
            trainer.train(dataset="unused.jsonl", steps=1)
        load_model.assert_not_called()


# ---------------------------------------------------------------------------
# CLI --report-to
# ---------------------------------------------------------------------------


class TestCliReportToFlag:
    @pytest.fixture
    def seen_train(self, monkeypatch):
        seen: dict = {}

        class Wide:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def train(self, **kw):
                return SimpleNamespace(final_loss=1.0, duration_seconds=1.0)

            def save(self, out):
                return out

        monkeypatch.setattr("backpropagate.trainer.Trainer", Wide)
        return seen

    @pytest.fixture
    def seen_multi(self, monkeypatch):
        seen: dict = {}

        class WideMulti:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def run(self, data):
                return SimpleNamespace(
                    total_runs=1, final_loss=1.0, total_duration_seconds=1.0,
                    final_checkpoint_path=None,
                )

        monkeypatch.setattr("backpropagate.multi_run.MultiRunTrainer", WideMulti)
        return seen

    def test_train_default_is_auto(self, seen_train):
        assert cli.cmd_train(parse(["train", "--data", "d"])) == cli.EXIT_OK
        assert seen_train["report_to"] == "auto"

    @pytest.mark.parametrize("value", ["none", "wandb", "tensorboard", "mlflow"])
    def test_train_flag_reaches_trainer(self, seen_train, value):
        assert cli.cmd_train(parse(["train", "--data", "d", "--report-to", value])) == cli.EXIT_OK
        assert seen_train["report_to"] == value

    def test_multi_run_flag_reaches_trainer(self, seen_multi):
        argv = ["multi-run", "--data", "d", "--report-to", "none"]
        assert cli.cmd_multi_run(parse(argv)) == cli.EXIT_OK
        assert seen_multi["report_to"] == "none"

    def test_multi_run_default_is_auto(self, seen_multi):
        assert cli.cmd_multi_run(parse(["multi-run", "--data", "d"])) == cli.EXIT_OK
        assert seen_multi["report_to"] == "auto"

    def test_real_multi_run_trainer_accepts_report_to(self):
        # The CLI routes report_to by introspection; it is only forwarded if
        # the real MultiRunTrainer still takes it.
        import inspect

        from backpropagate.multi_run import MultiRunTrainer

        assert "report_to" in inspect.signature(MultiRunTrainer.__init__).parameters
        assert "report_to" in inspect.signature(trainer_mod.Trainer.__init__).parameters

    @pytest.mark.parametrize("cmd", ["train", "multi-run"])
    def test_unknown_tracker_rejected_by_argparse(self, cmd):
        with pytest.raises(SystemExit) as excinfo:
            parse([cmd, "--data", "d", "--report-to", "comet"])
        assert excinfo.value.code == 2

    def test_explicit_wandb_unconfigured_exits_with_clear_message(self, monkeypatch, capsys):
        real_trainer = trainer_mod.Trainer

        class Raising:
            def __init__(self, *a, **kw):
                self.report_to = kw["report_to"]

            def train(self, **kw):
                # Real resolver against the real intent the CLI forwarded.
                t = real_trainer.__new__(real_trainer)
                t._report_to_intent = self.report_to
                t._wandb_skip_logged = False
                t._resolve_report_to()

        monkeypatch.setattr("backpropagate.trainer.Trainer", Raising)
        with _features(wandb=True):
            code = cli.cmd_train(parse(["train", "--data", "d", "--report-to", "wandb"]))
        assert code == cli.EXIT_RUNTIME_ERROR
        out = capsys.readouterr()
        text = out.out + out.err
        assert "CONFIG_INVALID_SETTING" in text
        assert "wandb login" in text
