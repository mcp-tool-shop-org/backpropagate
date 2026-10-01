"""Regression: ``BACKPROPAGATE_DEFER_FEATURE_DETECTION`` must be recoverable.

With the variable set, the import-time probe is skipped and every flag stays
``False``. The documented recovery (``refresh_features()``, and the
auto-refresh on the first ``require_feature`` / ``ensure_feature`` miss) used
to re-run the same probe, which honoured the variable AGAIN and returned
immediately, so an installed extra still looked missing and the operator got
the misleading "pip install backpropagate[...]" hint (BRIDGE-B-014).

Mocked: ``feature_flags._has_module`` (which optional packages "exist").
"""

from __future__ import annotations

import logging

import pytest

from backpropagate import feature_flags as ff

DEFER = "BACKPROPAGATE_DEFER_FEATURE_DETECTION"
QUIET = "BACKPROPAGATE_DEFER_FEATURE_QUIET"


@pytest.fixture
def pristine_features(monkeypatch):
    """Snapshot/restore the module-level FEATURES dict and the env switches."""
    snapshot = dict(ff.FEATURES)
    monkeypatch.delenv(DEFER, raising=False)
    monkeypatch.delenv(QUIET, raising=False)
    yield
    ff.FEATURES.clear()
    ff.FEATURES.update(snapshot)


def _installed(monkeypatch, *names: str) -> None:
    monkeypatch.setattr(ff, "_has_module", lambda name: name in names)


def _all_false() -> None:
    for key in ff.FEATURES:
        ff.FEATURES[key] = False


class TestImportTimeDeferral:
    def test_deferral_leaves_every_flag_false_and_warns(self, pristine_features, monkeypatch, caplog):
        _installed(monkeypatch, "pydantic", "pydantic_settings", "reflex")
        monkeypatch.setenv(DEFER, "1")
        _all_false()
        with caplog.at_level(logging.WARNING, logger="backpropagate.feature_flags"):
            ff._detect_features()
        assert not any(ff.FEATURES.values())
        assert any("Feature detection deferred" in r.getMessage() and r.levelno == logging.WARNING
                   for r in caplog.records)

    def test_quiet_mode_downgrades_the_warning_to_debug(self, pristine_features, monkeypatch, caplog):
        monkeypatch.setenv(DEFER, "1")
        monkeypatch.setenv(QUIET, "1")
        _all_false()
        with caplog.at_level(logging.DEBUG, logger="backpropagate.feature_flags"):
            ff._detect_features()
        deferred = [r for r in caplog.records if "deferred" in r.getMessage()]
        assert deferred and all(r.levelno == logging.DEBUG for r in deferred)
        assert any("quiet mode" in r.getMessage() for r in deferred)


class TestRefreshWhileDeferred:
    def test_refresh_features_detects_even_though_the_variable_is_still_set(
        self, pristine_features, monkeypatch
    ):
        _installed(monkeypatch, "pydantic", "pydantic_settings", "reflex", "psutil", "wandb")
        monkeypatch.setenv(DEFER, "1")
        result = ff.refresh_features()
        assert result["validation"] is True
        assert result["ui"] is True
        assert result["psutil"] is True and result["monitoring"] is True
        assert result["unsloth"] is False  # genuinely not "installed" in this scenario
        assert result == ff.FEATURES

    def test_refresh_returns_a_copy(self, pristine_features, monkeypatch):
        _installed(monkeypatch)
        result = ff.refresh_features()
        result["ui"] = True
        assert ff.FEATURES["ui"] is False


class TestAutoRefreshOnFirstUse:
    def test_ensure_feature_recovers_when_the_extra_is_installed(self, pristine_features, monkeypatch):
        _installed(monkeypatch, "pydantic", "pydantic_settings")
        monkeypatch.setenv(DEFER, "1")
        _all_false()
        ff.ensure_feature("validation")  # must NOT raise: it is installed
        assert ff.FEATURES["validation"] is True

    def test_require_feature_recovers_and_runs_the_function(self, pristine_features, monkeypatch):
        _installed(monkeypatch, "reflex")
        monkeypatch.setenv(DEFER, "1")
        _all_false()

        @ff.require_feature("ui")
        def launch(x):
            return x * 2

        assert launch(21) == 42
        assert ff.FEATURES["ui"] is True

    def test_a_genuinely_missing_feature_still_gets_the_install_hint(self, pristine_features, monkeypatch):
        _installed(monkeypatch, "reflex")  # ui yes, mlx no
        monkeypatch.setenv(DEFER, "1")
        _all_false()
        with pytest.raises(ff.FeatureNotAvailable) as exc:
            ff.ensure_feature("mlx")
        assert exc.value.feature == "mlx"
        assert "backpropagate[mlx]" in exc.value.install_hint
        assert ff.FEATURES["ui"] is True  # the auto-refresh did run

    def test_no_refresh_without_the_variable(self, pristine_features, monkeypatch):
        calls = []
        monkeypatch.setattr(ff, "refresh_features", lambda: calls.append(1))
        _all_false()
        ff._maybe_refresh_for_deferral("ui")
        assert calls == []

    def test_no_second_refresh_once_something_is_detected(self, pristine_features, monkeypatch):
        calls = []
        monkeypatch.setattr(ff, "refresh_features", lambda: calls.append(1))
        monkeypatch.setenv(DEFER, "1")
        _all_false()
        ff.FEATURES["triton"] = True
        ff._maybe_refresh_for_deferral("ui")
        assert calls == []
        _all_false()
        ff._maybe_refresh_for_deferral("ui")
        assert calls == [1]
