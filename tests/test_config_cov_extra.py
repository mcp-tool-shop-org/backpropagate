"""Coverage tests for ``backpropagate.config``.

Windows env-fix apply/unapply bookkeeping, the deprecated-env-var scan, the
package-version fallback, security helpers, validators, model presets and the
dataclass fallback branch (reached by executing ``config.py`` with
``pydantic_settings`` made un-importable, exactly as ``test_config_env_fallback``
does, so the executed lines are attributed to ``config.py``).

Mocked (and nothing else): ``backpropagate.logging_config.get_logger`` (to
capture structured events or force the stdlib-logger fallback),
``config._pkg_version`` (package metadata lookup), and a tiny ``os`` proxy
(``name`` + the real ``environ``) to run the Windows branch on any OS.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import pytest

import backpropagate.config as cfg
from backpropagate.exceptions import InvalidSettingError

ENV_KEYS = ("TOKENIZERS_PARALLELISM", "XFORMERS_DISABLED", "CUDA_LAUNCH_BLOCKING")


class _OS:
    """Minimal stand-in for ``os`` inside config.py (it only touches name/environ)."""

    def __init__(self, name: str):
        self.name = name
        self.environ = os.environ


class _Recorder:
    """A structlog-ish logger that remembers every call."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def _rec(self, level):
        def record(*args, **kwargs):
            self.calls.append((level, args, kwargs))

        return record

    def __getattr__(self, name):
        if name in {"debug", "info", "warning", "error"}:
            return self._rec(name)
        raise AttributeError(name)


@pytest.fixture
def isolated_env(monkeypatch):
    """Snapshot/restore the windows-fix bookkeeping and the env vars it touches."""
    saved_applied = dict(cfg._WINDOWS_FIXES_APPLIED)
    cfg._WINDOWS_FIXES_APPLIED.clear()
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ):
        if key.upper().startswith("BACKPROPAGATE_"):
            monkeypatch.delenv(key, raising=False)
    yield
    cfg._WINDOWS_FIXES_APPLIED.clear()
    cfg._WINDOWS_FIXES_APPLIED.update(saved_applied)


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("backpropagate.logging_config.get_logger", lambda name=None: rec)
    return rec


@pytest.fixture
def broken_structured_logging(monkeypatch):
    def boom(name=None):
        raise RuntimeError("logging not configured")

    monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)


# =============================================================================
# _safe_pkg_version
# =============================================================================


class TestSafePkgVersion:
    def test_returns_installed_version(self, monkeypatch):
        monkeypatch.setattr(cfg, "_pkg_version", lambda name: "9.8.7")
        assert cfg._safe_pkg_version() == "9.8.7"

    def test_falls_back_to_local_sentinel_when_metadata_is_missing(self, monkeypatch):
        def missing(name):
            raise PackageNotFoundError(name)

        monkeypatch.setattr(cfg, "_pkg_version", missing)
        assert cfg._safe_pkg_version() == "0.0.0+unknown"


# =============================================================================
# Deprecated env-var scan
# =============================================================================


class TestDeprecatedEnvScan:
    def test_nothing_set_returns_empty(self, isolated_env):
        assert cfg._warn_deprecated_env_vars() == []

    def test_renamed_knob_warns_with_the_replacement(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setenv("BACKPROPAGATE_MULTI_RUN__NUM_RUNS", "3")
        found = cfg._warn_deprecated_env_vars()
        assert found == ["BACKPROPAGATE_MULTI_RUN__NUM_RUNS"]
        (level, args, _), = recorder.calls
        assert level == "warning"
        assert args[1] == "BACKPROPAGATE_MULTI_RUN__NUM_RUNS"
        assert args[2] == "BACKPROPAGATE_MULTIRUN__NUM_RUNS"
        assert "no longer read" in args[0]

    def test_all_known_renames_are_detected(self, isolated_env, monkeypatch, recorder):
        for old in cfg._DEPRECATED_ENV_VARS:
            monkeypatch.setenv(old, "1")
        assert set(cfg._warn_deprecated_env_vars()) == set(cfg._DEPRECATED_ENV_VARS)
        assert len(recorder.calls) == len(cfg._DEPRECATED_ENV_VARS)

    def test_removed_knob_says_there_is_no_replacement(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setitem(cfg._DEPRECATED_ENV_VARS, "BACKPROPAGATE_GONE__KNOB", None)
        monkeypatch.setenv("BACKPROPAGATE_GONE__KNOB", "x")
        assert cfg._warn_deprecated_env_vars() == ["BACKPROPAGATE_GONE__KNOB"]
        (level, args, _), = recorder.calls
        assert "has been removed" in args[0] and args[1] == "BACKPROPAGATE_GONE__KNOB"

    def test_falls_back_to_stdlib_logging_when_structured_logging_breaks(
        self, isolated_env, monkeypatch, broken_structured_logging, caplog
    ):
        monkeypatch.setenv("BACKPROPAGATE_MULTI_RUN__STEPS_PER_RUN", "10")
        with caplog.at_level(logging.WARNING, logger="backpropagate.config"):
            found = cfg._warn_deprecated_env_vars()
        assert found == ["BACKPROPAGATE_MULTI_RUN__STEPS_PER_RUN"]
        assert any(
            "BACKPROPAGATE_MULTIRUN__STEPS_PER_RUN" in r.getMessage() for r in caplog.records
        )

    def test_get_settings_runs_the_scan_once_per_cache_lifetime(self, isolated_env, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_MULTI_RUN__NUM_RUNS", "2")
        cfg.get_settings.cache_clear()
        try:
            with patch.object(cfg, "_warn_deprecated_env_vars", wraps=cfg._warn_deprecated_env_vars) as scan:
                cfg.get_settings()
                cfg.get_settings()
            assert scan.call_count == 1
        finally:
            monkeypatch.delenv("BACKPROPAGATE_MULTI_RUN__NUM_RUNS")
            cfg.reload_settings()


# =============================================================================
# apply / unapply windows fixes
# =============================================================================


class TestEnvMutationBookkeeping:
    def test_apply_records_prior_values_and_unapply_restores_them(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setenv("XFORMERS_DISABLED", "original")
        cfg._apply_env_mutations({"XFORMERS_DISABLED": "1", "TOKENIZERS_PARALLELISM": "false"})
        assert os.environ["XFORMERS_DISABLED"] == "1"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert cfg._WINDOWS_FIXES_APPLIED["XFORMERS_DISABLED"] == ("original", "1")
        assert isinstance(cfg._WINDOWS_FIXES_APPLIED["TOKENIZERS_PARALLELISM"][0], cfg._Unset)
        (level, args, kwargs), = [c for c in recorder.calls if c[1] == ("windows_fixes_applied",)]
        assert set(kwargs["mutated"]) == {"XFORMERS_DISABLED", "TOKENIZERS_PARALLELISM"}
        assert kwargs["values"]["XFORMERS_DISABLED"] == "1"

        cfg.unapply_windows_fixes()
        assert os.environ["XFORMERS_DISABLED"] == "original"  # restored, not deleted
        assert "TOKENIZERS_PARALLELISM" not in os.environ  # was unset before -> popped
        assert cfg._WINDOWS_FIXES_APPLIED == {}
        assert any(c[1] == ("windows_fixes_unapplied",) for c in recorder.calls)

    def test_second_apply_keeps_the_first_prior_value(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setenv("XFORMERS_DISABLED", "pristine")
        cfg._apply_env_mutations({"XFORMERS_DISABLED": "1"})
        cfg._apply_env_mutations({"XFORMERS_DISABLED": "2"})
        assert cfg._WINDOWS_FIXES_APPLIED["XFORMERS_DISABLED"] == ("pristine", "2")
        cfg.unapply_windows_fixes()
        assert os.environ["XFORMERS_DISABLED"] == "pristine"

    def test_unapply_twice_is_a_noop(self, isolated_env, recorder):
        cfg._apply_env_mutations({"CUDA_LAUNCH_BLOCKING": "1"})
        cfg.unapply_windows_fixes()
        cfg.unapply_windows_fixes()
        assert "CUDA_LAUNCH_BLOCKING" not in os.environ

    def test_empty_string_prior_is_restored_as_empty_not_unset(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "")
        cfg._apply_env_mutations({"TOKENIZERS_PARALLELISM": "false"})
        cfg.unapply_windows_fixes()
        assert os.environ["TOKENIZERS_PARALLELISM"] == ""

    def test_mutations_apply_even_when_structured_logging_is_broken(
        self, isolated_env, broken_structured_logging
    ):
        cfg._apply_env_mutations({"XFORMERS_DISABLED": "1"})
        assert os.environ["XFORMERS_DISABLED"] == "1"
        cfg.unapply_windows_fixes()  # observability failure is swallowed here too
        assert "XFORMERS_DISABLED" not in os.environ


class TestApplyWindowsFixes:
    """Runs on any OS by swapping config.py's view of ``os`` for a proxy."""

    def test_windows_defaults_set_parallelism_and_xformers(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setattr(cfg, "os", _OS("nt"))
        settings = cfg.Settings()
        settings.apply_windows_fixes()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert os.environ["XFORMERS_DISABLED"] == "1"
        assert "CUDA_LAUNCH_BLOCKING" not in os.environ
        cfg.unapply_windows_fixes()
        assert "XFORMERS_DISABLED" not in os.environ

    def test_launch_blocking_and_parallelism_follow_settings(self, isolated_env, monkeypatch, recorder):
        monkeypatch.setattr(cfg, "os", _OS("nt"))
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__CUDA_LAUNCH_BLOCKING", "true")
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__TOKENIZERS_PARALLELISM", "true")
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED", "false")
        cfg.Settings().apply_windows_fixes()
        assert os.environ["CUDA_LAUNCH_BLOCKING"] == "1"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
        assert "XFORMERS_DISABLED" not in os.environ
        cfg.unapply_windows_fixes()

    def test_posix_hosts_are_left_alone(self, isolated_env, monkeypatch):
        monkeypatch.setattr(cfg, "os", _OS("posix"))
        cfg.Settings().apply_windows_fixes()
        assert not any(k in os.environ for k in ENV_KEYS)
        assert cfg._WINDOWS_FIXES_APPLIED == {}

    def test_training_args_use_windows_dataloader_workers_only_on_nt(self, monkeypatch):
        monkeypatch.setattr(cfg, "os", _OS("nt"))
        assert cfg.get_training_args()["dataloader_num_workers"] == cfg.settings.windows.dataloader_num_workers
        monkeypatch.setattr(cfg, "os", _OS("posix"))
        assert cfg.get_training_args()["dataloader_num_workers"] == 4


# =============================================================================
# Validators
# =============================================================================


class TestTrainingValidators:
    def test_invalid_backend_kwarg_and_env(self, isolated_env, monkeypatch):
        with pytest.raises(InvalidSettingError) as exc:
            cfg.TrainingConfig(backend="rocm")
        assert exc.value.code == "CONFIG_INVALID_SETTING"
        assert exc.value.setting_name == "backend"
        assert "'auto', 'cuda', 'mlx'" in exc.value.expected
        monkeypatch.setenv("BACKPROPAGATE_TRAINING__BACKEND", "tpu")
        with pytest.raises(InvalidSettingError, match="tpu"):
            cfg.TrainingConfig()

    @pytest.mark.parametrize("backend", ["auto", "cuda", "mlx"])
    def test_valid_backends(self, backend):
        assert cfg.TrainingConfig(backend=backend).backend == backend

    def test_high_simpo_gamma_ratio_warns_but_constructs(self, isolated_env, recorder):
        t = cfg.TrainingConfig(simpo_beta=1.0, simpo_gamma=3.0)
        assert t.simpo_gamma == 3.0
        warns = [c for c in recorder.calls if c[0] == "warning"]
        assert len(warns) == 1
        assert warns[0][1][3] == 3.0  # the ratio is reported
        assert "degeneration band" in warns[0][1][0]

    def test_high_gamma_ratio_with_broken_structured_logging_uses_stdlib(
        self, isolated_env, broken_structured_logging, caplog
    ):
        with caplog.at_level(logging.WARNING, logger="backpropagate.config"):
            cfg.TrainingConfig(simpo_beta=1.0, simpo_gamma=2.0)
        assert any("degeneration band" in r.getMessage() for r in caplog.records)

    def test_default_simpo_settings_do_not_warn(self, isolated_env, recorder):
        cfg.TrainingConfig()
        assert recorder.calls == []

    def test_non_positive_validators_name_the_setting(self, isolated_env):
        for kwargs, name in [
            ({"orpo_beta": 0}, "orpo_beta"),
            ({"simpo_gamma": -1}, "simpo_gamma"),
            ({"kto_desirable_weight": 0}, "kto_desirable_weight"),
            ({"kto_undesirable_weight": -2}, "kto_undesirable_weight"),
            ({"method": "dpo"}, "method"),
            ({"bf16": True, "fp16": True}, "bf16/fp16"),
        ]:
            with pytest.raises(InvalidSettingError) as exc:
                cfg.TrainingConfig(**kwargs)
            assert exc.value.setting_name == name, kwargs
            assert exc.value.code == "CONFIG_INVALID_SETTING"


class TestSecurityConfigHelpers:
    def test_get_auth_tuple_requires_both_credentials(self, isolated_env):
        assert cfg.SecurityConfig().get_auth_tuple() is None
        assert cfg.SecurityConfig(auth_username="u").get_auth_tuple() is None
        assert cfg.SecurityConfig(auth_password="p").get_auth_tuple() is None
        assert cfg.SecurityConfig(auth_username="u", auth_password="p").get_auth_tuple() == ("u", "p")

    def test_default_config_lists_the_production_gaps(self, isolated_env):
        warnings = cfg.SecurityConfig().validate_production_config()
        assert any("require_auth is False" in w for w in warnings)
        assert any("jwt_secret not set" in w for w in warnings)
        assert not any("CSRF" in w for w in warnings)
        assert not any("session_timeout" in w for w in warnings)

    def test_hardened_config_has_no_warnings(self, isolated_env):
        c = cfg.SecurityConfig(require_auth=True, jwt_secret="s3cret", enable_csrf=True, session_timeout_minutes=30)
        assert c.validate_production_config() == []

    def test_csrf_and_long_sessions_are_flagged(self, isolated_env):
        c = cfg.SecurityConfig(
            require_auth=True, jwt_secret="s", enable_csrf=False, session_timeout_minutes=481
        )
        warnings = c.validate_production_config()
        assert any("CSRF protection disabled" in w for w in warnings)
        assert any("session_timeout_minutes > 8 hours" in w for w in warnings)
        boundary = cfg.SecurityConfig(require_auth=True, jwt_secret="s", session_timeout_minutes=480)
        assert boundary.validate_production_config() == []

    def test_settings_to_dict_summary(self, isolated_env):
        d = cfg.Settings().to_dict()
        assert set(d) == {"version", "model", "training", "lora", "data"}
        assert d["training"]["batch_size"] == 2 and d["lora"]["r"] == 256
        assert d["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
        assert d["data"]["max_samples"] == 0


# =============================================================================
# Presets / helpers
# =============================================================================


class TestPresetLookups:
    def test_get_model_preset_by_name(self):
        p = cfg.get_model_preset("qwen2.5-14b")
        assert p.model_id == "Qwen/Qwen2.5-14B-Instruct"
        assert p.license == "Apache-2.0" and p.recommended_lora_r == 32

    def test_unknown_model_preset_lists_the_catalog(self):
        with pytest.raises(ValueError) as exc:
            cfg.get_model_preset("gpt-9")
        assert "Unknown model preset 'gpt-9'" in str(exc.value)
        for name in cfg.MODEL_PRESETS:
            assert name in str(exc.value)

    def test_lookup_by_model_id_is_case_insensitive_and_trims(self):
        p = cfg.lookup_model_preset_by_id("  qwen/qwen2.5-3b-instruct ")
        assert p is not None and p.name == "qwen2.5-3b"
        assert p.license_restriction and "non-commercial" in p.license_restriction
        assert cfg.lookup_model_preset_by_id("nobody/nothing") is None

    def test_lora_and_training_preset_errors(self):
        with pytest.raises(ValueError, match="Unknown LoRA preset 'turbo'. Available: fast, balanced, quality"):
            cfg.get_lora_preset("turbo")
        with pytest.raises(ValueError, match="Unknown preset 'warp'. Available: "):
            cfg.get_preset("warp")
        assert cfg.get_lora_preset("fast").r == 16
        assert cfg.get_preset("balanced").effective_batch_size == 16

    def test_every_model_preset_is_self_consistent(self):
        for key, preset in cfg.MODEL_PRESETS.items():
            assert preset.name == key
            assert cfg.lookup_model_preset_by_id(preset.model_id) is preset
            assert preset.recommended_lora_r > 0 and preset.recommended_max_seq_length >= 2048

    @pytest.mark.parametrize(
        ("size", "lr", "warmup_ratio"),
        [(10, 5e-4, 0.15), (999, 5e-4, 0.15), (1000, 2e-4, 0.10), (9999, 2e-4, 0.10), (10000, 1e-4, 0.05)],
    )
    def test_lr_and_warmup_ladders_at_the_boundaries(self, size, lr, warmup_ratio):
        assert cfg.get_recommended_lr(size) == pytest.approx(lr)
        assert cfg.get_recommended_warmup(size, 200) == int(200 * warmup_ratio)

    def test_lr_scaling_and_method_anchors(self):
        assert cfg.get_recommended_lr(500, base_lr=4e-4) == pytest.approx(1e-3)
        assert cfg.get_recommended_lr(500, method="orpo") == 2e-5
        assert cfg.get_recommended_lr(5000, method="orpo") == 1e-5
        assert cfg.get_recommended_lr(50000, method="orpo") == 5e-6
        assert cfg.get_recommended_lr(1, method="simpo") == cfg.get_recommended_lr(10**6, method="kto") == 1e-6
        assert cfg.get_recommended_warmup(100, 3) == 1  # never below one step

    def test_output_and_cache_dirs_are_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg.settings.training, "output_dir", str(tmp_path / "out" / "deep"))
        assert cfg.get_output_dir() == tmp_path / "out" / "deep" and (tmp_path / "out" / "deep").is_dir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert cfg.get_cache_dir() == tmp_path / ".cache" / "backpropagate"
        assert cfg.get_cache_dir().is_dir()


# =============================================================================
# Dataclass fallback branch (pydantic_settings blocked)
# =============================================================================


def _load_fallback() -> types.ModuleType:
    sys.modules.setdefault("backpropagate", importlib.import_module("backpropagate"))
    source = Path(cfg.__file__).read_text(encoding="utf-8")
    fake = types.ModuleType("backpropagate._config_cov_probe")
    fake.__dict__["__file__"] = cfg.__file__
    fake.__dict__["__name__"] = "backpropagate.config"
    with patch.dict(sys.modules, {"pydantic_settings": None}):
        exec(compile(source, cfg.__file__, "exec"), fake.__dict__)  # noqa: S102
    assert fake.__dict__["PYDANTIC_SETTINGS_AVAILABLE"] is False
    return fake


@pytest.fixture
def fb(isolated_env):
    return _load_fallback()


class TestFallbackBranch:
    def test_integral_float_text_is_accepted_for_int_fields(self, fb, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH", "4096.0")
        assert fb.ModelConfig().max_seq_length == 4096

    def test_fractional_float_text_is_rejected_for_int_fields(self, fb, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_MODEL__MAX_SEQ_LENGTH", "4096.5")
        with pytest.raises(InvalidSettingError, match="an integer"):
            fb.ModelConfig()

    def test_invalid_backend_is_rejected(self, fb):
        with pytest.raises(InvalidSettingError) as exc:
            fb.TrainingConfig(backend="rocm")
        assert exc.value.setting_name == "backend"
        assert fb.TrainingConfig(backend="mlx").backend == "mlx"

    def test_other_post_init_validators(self, fb):
        for kwargs, name in [
            ({"orpo_beta": 0}, "orpo_beta"),
            ({"simpo_gamma": 0}, "simpo_gamma"),
            ({"kto_desirable_weight": -1}, "kto_desirable_weight"),
            ({"kto_undesirable_weight": 0}, "kto_undesirable_weight"),
            ({"method": "ppo"}, "method"),
            ({"bf16": True, "fp16": True}, "bf16/fp16"),
        ]:
            with pytest.raises(InvalidSettingError) as exc:
                fb.TrainingConfig(**kwargs)
            assert exc.value.setting_name == name

    def test_high_simpo_gamma_ratio_warns(self, fb, recorder):
        fb_t = fb.TrainingConfig(simpo_beta=1.0, simpo_gamma=4.0)
        assert fb_t.simpo_gamma == 4.0
        warns = [c for c in recorder.calls if c[0] == "warning"]
        assert len(warns) == 1 and warns[0][1][3] == 4.0

    def test_high_simpo_gamma_ratio_with_broken_structured_logging(self, fb, broken_structured_logging, caplog):
        with caplog.at_level(logging.WARNING, logger="backpropagate.config"):
            fb.TrainingConfig(simpo_beta=1.0, simpo_gamma=2.0)
        assert any("degeneration band" in r.getMessage() for r in caplog.records)

    def test_security_helpers(self, fb):
        assert fb.SecurityConfig().get_auth_tuple() is None
        assert fb.SecurityConfig(auth_username="u").get_auth_tuple() is None
        assert fb.SecurityConfig(auth_username="u", auth_password="p").get_auth_tuple() == ("u", "p")
        warnings = fb.SecurityConfig().validate_production_config()
        assert warnings == ["SECURITY: require_auth is False", "SECURITY: jwt_secret not set"]
        assert fb.SecurityConfig(require_auth=True, jwt_secret="s").validate_production_config() == []

    def test_settings_to_dict_is_minimal(self, fb):
        assert fb.Settings().to_dict() == {"version": "0.1.0"}

    def test_windows_fixes_apply_through_the_shared_bookkeeping(self, fb, monkeypatch, recorder):
        monkeypatch.setattr(fb, "os", _OS("nt"))
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__CUDA_LAUNCH_BLOCKING", "1")
        fb.Settings().apply_windows_fixes()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert os.environ["XFORMERS_DISABLED"] == "1"
        assert os.environ["CUDA_LAUNCH_BLOCKING"] == "1"
        assert set(fb._WINDOWS_FIXES_APPLIED) == {
            "TOKENIZERS_PARALLELISM", "XFORMERS_DISABLED", "CUDA_LAUNCH_BLOCKING",
        }
        fb.unapply_windows_fixes()
        assert not any(k in os.environ for k in ENV_KEYS)

    def test_windows_fixes_skip_optional_flags_and_non_windows(self, fb, monkeypatch, recorder):
        monkeypatch.setattr(fb, "os", _OS("nt"))
        monkeypatch.setenv("BACKPROPAGATE_WINDOWS__XFORMERS_DISABLED", "0")
        fb.Settings().apply_windows_fixes()
        assert "XFORMERS_DISABLED" not in os.environ and "CUDA_LAUNCH_BLOCKING" not in os.environ
        fb.unapply_windows_fixes()
        monkeypatch.setattr(fb, "os", _OS("posix"))
        fb.Settings().apply_windows_fixes()
        assert not any(k in os.environ for k in ENV_KEYS)
