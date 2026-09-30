"""BACKPROPAGATE_* configuration variables must work on every install.

``pydantic`` / ``pydantic-settings`` are optional (``[validation]`` extra). A
plain ``pip install backpropagate`` therefore runs the dataclass fallback in
``config.py``, which used to ignore every documented ``BACKPROPAGATE_<GROUP>__
<FIELD>`` variable. These tests pin two things:

* the environment variables take effect in whichever branch this interpreter
  runs (the "normal CI environment" tests), and
* with ``pydantic_settings`` made un-importable the fallback branch still
  honours them -- bool / int / float / str / list -- rejects malformed values
  loudly, and agrees with the pydantic branch field by field (parity).
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

import backpropagate.config as real_cfg
from backpropagate.exceptions import InvalidSettingError

SECTION_CLASSES = {
    "MODEL": "ModelConfig",
    "LORA": "LoRAConfig",
    "TRAINING": "TrainingConfig",
    "DATA": "DataConfig",
    "UI": "UIConfig",
    "WINDOWS": "WindowsConfig",
    "MULTIRUN": "MultiRunSettings",
    "SECURITY": "SecurityConfig",
}


def _load_fallback_module() -> types.ModuleType:
    """Exec config.py with pydantic_settings blocked -> dataclass branch."""
    sys.modules.setdefault("backpropagate", importlib.import_module("backpropagate"))
    source = Path(real_cfg.__file__).read_text(encoding="utf-8")
    fake = types.ModuleType("backpropagate._config_env_fallback_probe")
    fake.__dict__["__file__"] = real_cfg.__file__
    fake.__dict__["__name__"] = "backpropagate.config"
    with patch.dict(sys.modules, {"pydantic_settings": None}):
        exec(compile(source, real_cfg.__file__, "exec"), fake.__dict__)  # noqa: S102
    assert fake.__dict__["PYDANTIC_SETTINGS_AVAILABLE"] is False, (
        "blocking pydantic_settings must select the dataclass fallback"
    )
    return fake


@pytest.fixture
def fallback() -> types.ModuleType:
    return _load_fallback_module()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Drop ambient BACKPROPAGATE_* config so each test starts from defaults."""
    for key in list(os.environ):
        if key.upper().startswith("BACKPROPAGATE_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def restore_settings():
    yield
    real_cfg.reload_settings()


# ---------------------------------------------------------------------------
# Normal environment: whichever branch this interpreter runs honours the env.
# ---------------------------------------------------------------------------


class TestEnvVarsHonouredInActiveBranch:
    def test_documented_variables_reach_settings(self, monkeypatch, restore_settings):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "false")
        monkeypatch.setenv("BACKPROPAGATE_MODEL__NAME", "org/some-model")
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__LEARNING_RATE", "0.123")
        monkeypatch.setenv("BACKPROPAGATE_LORA__R", "32")
        monkeypatch.setenv("BACKPROPAGATE_MULTIRUN__NUM_RUNS", "9")
        monkeypatch.setenv("BACKPROPAGATE_DATA__PACKING", "false")
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED", "0")

        s = real_cfg.reload_settings()

        assert s.model.trust_remote_code is False
        assert s.model.name == "org/some-model"
        assert s.training.learning_rate == pytest.approx(0.123)
        assert s.lora.r == 32
        assert s.multi_run.num_runs == 9
        assert s.data.packing is False
        assert s.windows.xformers_disabled is False

    def test_invalid_value_is_an_error_not_a_silent_default(self, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__LEARNING_RATE", "fast")
        with pytest.raises(Exception) as exc:  # noqa: PT011 - branch-dependent type
            real_cfg.TrainingConfig()
        assert "learning_rate" in str(exc.value).lower() or "LEARNING_RATE" in str(exc.value)


# ---------------------------------------------------------------------------
# Dataclass fallback (pydantic-settings absent)
# ---------------------------------------------------------------------------


class TestFallbackHonoursEnv:
    def test_bool_float_int_str(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "false")
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__LEARNING_RATE", "0.123")
        monkeypatch.setenv("BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH", "4096")
        monkeypatch.setenv("BACKPROPAGATE_MODEL__NAME", "org/some-model")

        model = fallback.ModelConfig()
        training = fallback.TrainingConfig()

        assert model.trust_remote_code is False
        assert model.max_seq_length == 4096
        assert model.name == "org/some-model"
        assert training.learning_rate == pytest.approx(0.123)

    def test_settings_container_sees_every_section(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "false")
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__LEARNING_RATE", "0.123")
        monkeypatch.setenv("BACKPROPAGATE_LORA__R", "32")
        monkeypatch.setenv("BACKPROPAGATE_DATA__MAX_SAMPLES", "7")
        monkeypatch.setenv("BACKPROPAGATE_MULTIRUN__NUM_RUNS", "9")
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__SESSION_TIMEOUT_MINUTES", "5")
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED", "no")
        monkeypatch.setenv("BACKPROPAGATE_UI__PORT", "9000")

        s = fallback.Settings()

        assert s.model.trust_remote_code is False
        assert s.training.learning_rate == pytest.approx(0.123)
        assert s.lora.r == 32
        assert s.data.max_samples == 7
        assert s.multi_run.num_runs == 9
        assert s.security.session_timeout_minutes == 5
        assert s.windows.xformers_disabled is False
        assert s.ui.port == 9000

    def test_module_level_settings_singleton_and_reload(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "false")
        assert fallback.reload_settings().model.trust_remote_code is False

    def test_list_fields_take_json_or_plain_string(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_LORA__TARGET_MODULES", '["q_proj","v_proj"]')
        assert fallback.LoRAConfig().target_modules == ["q_proj", "v_proj"]
        monkeypatch.setenv("BACKPROPAGATE_LORA__TARGET_MODULES", "all-linear")
        assert fallback.LoRAConfig().target_modules == "all-linear"
        monkeypatch.setenv("BACKPROPAGATE_SECURITY__ALLOWED_PATHS", '["/data"]')
        assert fallback.SecurityConfig().allowed_paths == ["/data"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
         ("false", False), ("False", False), ("0", False), ("no", False), ("off", False)],
    )
    def test_bool_spellings(self, monkeypatch, fallback, raw, expected):
        monkeypatch.setenv("BACKPROPAGATE_DATA__SHUFFLE", raw)
        assert fallback.DataConfig().shuffle is expected

    def test_variable_name_is_case_insensitive(self, monkeypatch, fallback):
        monkeypatch.setenv("backpropagate_model__name", "org/lower")
        assert fallback.ModelConfig().name == "org/lower"

    def test_empty_value_is_ignored(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "")
        assert fallback.ModelConfig().trust_remote_code is fallback.ModelConfig.__dataclass_fields__[
            "trust_remote_code"
        ].default

    def test_explicit_argument_beats_environment(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__NAME", "from/env")
        assert fallback.ModelConfig(name="from/arg").name == "from/arg"
        assert fallback.ModelConfig("positional/arg").name == "positional/arg"

    def test_unset_env_keeps_defaults(self, fallback):
        assert fallback.TrainingConfig().learning_rate == pytest.approx(2e-4)

    def test_post_init_validation_still_applies_to_env_values(self, monkeypatch, fallback):
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__METHOD", "dpo")
        with pytest.raises(InvalidSettingError):
            fallback.TrainingConfig()
        monkeypatch.delenv("BACKPROPAGATE_TRAINING__METHOD")
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__FP16", "true")  # bf16 defaults true
        with pytest.raises(InvalidSettingError):
            fallback.TrainingConfig()


class TestFallbackRejectsBadValues:
    @pytest.mark.parametrize(
        ("var", "value", "cls"),
        [
            ("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "maybe", "ModelConfig"),
            ("BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH", "lots", "ModelConfig"),
            ("BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH", "12.5", "ModelConfig"),
            ("BACKPROPAGATE_TRAINING__LEARNING_RATE", "fast", "TrainingConfig"),
            ("BACKPROPAGATE_LORA__TARGET_MODULES", "[1, 2]", "LoRAConfig"),
            ("BACKPROPAGATE_SECURITY__ALLOWED_PATHS", "/data,/models", "SecurityConfig"),
        ],
    )
    def test_clear_structured_error(self, monkeypatch, fallback, var, value, cls):
        monkeypatch.setenv(var, value)
        with pytest.raises(InvalidSettingError) as exc:
            getattr(fallback, cls)()
        assert exc.value.code == "CONFIG_INVALID_SETTING"
        assert var in str(exc.value)  # names the variable the operator set
        assert value in str(exc.value)
        assert var in (exc.value.suggestion or "")


# ---------------------------------------------------------------------------
# Parity: fallback == pydantic branch, field by field
# ---------------------------------------------------------------------------


def _sample_env_value(field: dataclasses.Field) -> str:
    name, default = field.name, field.default
    if name == "method":
        return "orpo"
    if name == "backend":
        return "cuda"
    if name in ("target_modules", "allowed_paths"):
        return '["alpha","beta"]'
    if field.type is bool:
        return "false" if default is True else "true"
    if field.type is int:
        return str(int(default) + 7)
    if field.type is float:
        return "0.375"
    return "custom-value"


class TestParityWithPydanticBranch:
    @pytest.fixture
    def pydantic_cfg(self):
        pytest.importorskip("pydantic_settings")
        assert real_cfg.PYDANTIC_SETTINGS_AVAILABLE
        return real_cfg

    def test_same_field_names_per_section(self, pydantic_cfg, fallback):
        for cls_name in SECTION_CLASSES.values():
            fb_fields = {f.name for f in dataclasses.fields(getattr(fallback, cls_name))}
            pyd_fields = set(getattr(pydantic_cfg, cls_name).model_fields)
            assert fb_fields == pyd_fields, cls_name

    def test_every_field_reads_the_same_value(self, monkeypatch, pydantic_cfg, fallback):
        checked = 0
        for group, cls_name in SECTION_CLASSES.items():
            fb_cls = getattr(fallback, cls_name)
            pyd_cls = getattr(pydantic_cfg, cls_name)
            for f in dataclasses.fields(fb_cls):
                var = f"BACKPROPAGATE_{group}__{f.name.upper()}"
                monkeypatch.setenv(var, _sample_env_value(f))
                try:
                    try:
                        expected = getattr(pyd_cls(), f.name)
                        pyd_err = None
                    except Exception as exc:  # noqa: BLE001
                        expected, pyd_err = None, exc
                    try:
                        got = getattr(fb_cls(), f.name)
                        fb_err = None
                    except Exception as exc:  # noqa: BLE001
                        got, fb_err = None, exc
                finally:
                    monkeypatch.delenv(var)
                # Same accept/reject decision and, when accepted, same value.
                assert (pyd_err is None) == (fb_err is None), (var, pyd_err, fb_err)
                assert got == expected, var
                checked += 1
        assert checked >= 60  # guards against the loop silently covering nothing

    @pytest.mark.parametrize(
        ("var", "value", "cls"),
        [
            ("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE", "maybe", "ModelConfig"),
            ("BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH", "lots", "ModelConfig"),
            ("BACKPROPAGATE_TRAINING__LEARNING_RATE", "fast", "TrainingConfig"),
            ("BACKPROPAGATE_SECURITY__ALLOWED_PATHS", "/data,/models", "SecurityConfig"),
        ],
    )
    def test_both_branches_reject_the_same_bad_values(
        self, monkeypatch, pydantic_cfg, fallback, var, value, cls
    ):
        monkeypatch.setenv(var, value)
        with pytest.raises(Exception):  # noqa: B017, PT011 - pydantic ValidationError
            getattr(pydantic_cfg, cls)()
        with pytest.raises(InvalidSettingError):
            getattr(fallback, cls)()
