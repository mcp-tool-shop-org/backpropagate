"""BACKPROPAGATE_UI__OUTPUT_DIR is documented; setting it must not crash Settings.

Before the fix, the pydantic UIConfig had no ``output_dir`` field, so the
nested ``BACKPROPAGATE_UI__OUTPUT_DIR`` input was rejected as
``extra_forbidden`` and every command (``backprop ui``, ``train``, ...) died at
settings load.
"""

from __future__ import annotations

from backpropagate.config import Settings
from backpropagate.ui_security import get_ui_output_dir


def test_settings_accept_ui_output_dir(monkeypatch, tmp_path):
    target = tmp_path / "ui-out"
    monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(target))
    settings = Settings()
    assert settings.ui.output_dir == str(target)


def test_ui_output_dir_still_drives_the_sandbox(monkeypatch, tmp_path):
    target = tmp_path / "ui-out"
    monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(target))
    assert get_ui_output_dir() == target.resolve()


def test_default_matches_the_documented_sandbox(monkeypatch):
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    assert Settings().ui.output_dir == "~/.backpropagate/ui-outputs"
