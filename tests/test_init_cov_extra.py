"""Coverage tests for the import-time fallbacks in ``backpropagate/__init__.py``.

Two module-level fallbacks never run in a normal environment: the
``ui_security`` import guard (``[ui]`` extra absent) and the
package-metadata fallback for a source checkout that was never
``pip install``-ed. Both are exercised by executing the real ``__init__.py``
source into a scratch namespace (attributed to the real file for coverage)
with the corresponding import / metadata lookup made to fail.

Mocked: ``sys.modules['backpropagate.ui_security'] = None`` (simulates the
missing extra) and ``importlib.metadata.version`` (package metadata lookup).
The real ``backpropagate`` package object is untouched.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import pytest

import backpropagate

UI_SECURITY_NAMES = (
    "SecurityConfig",
    "DEFAULT_SECURITY_CONFIG",
    "EnhancedRateLimiter",
    "FileValidator",
    "ALLOWED_DATASET_EXTENSIONS",
    "DANGEROUS_EXTENSIONS",
    "log_security_event",
)


def _exec_init(monkeypatch, *, blocked: tuple[str, ...] = (), version_error: bool = False) -> dict:
    """Execute ``backpropagate/__init__.py`` into a fresh namespace."""
    # __init__ sets UNSLOTH_AUTO_INSTALL; restore whatever was there afterwards.
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", os.environ.get("UNSLOTH_AUTO_INSTALL", "0"))
    source = Path(backpropagate.__file__).read_text(encoding="utf-8")
    namespace: dict = {
        "__name__": "backpropagate",
        "__package__": "backpropagate",
        "__file__": backpropagate.__file__,
        "__path__": list(backpropagate.__path__),
    }
    patches: list = [patch.dict(sys.modules, dict.fromkeys(blocked))]
    if version_error:
        def missing(name):
            raise PackageNotFoundError(name)

        patches.append(patch.object(importlib.metadata, "version", missing))
    for p in patches:
        p.start()
    try:
        exec(compile(source, backpropagate.__file__, "exec"), namespace)  # noqa: S102
    finally:
        for p in reversed(patches):
            p.stop()
    return namespace


class TestUiSecurityGuard:
    def test_ui_security_helpers_resolve_when_the_module_imports(self, monkeypatch):
        ns = _exec_init(monkeypatch)
        for name in UI_SECURITY_NAMES:
            assert ns[name] is not None, name
            assert ns[name] is getattr(backpropagate, name)

    def test_missing_ui_security_sets_every_exported_name_to_none(self, monkeypatch):
        ns = _exec_init(monkeypatch, blocked=("backpropagate.ui_security",))
        for name in UI_SECURITY_NAMES:
            assert ns[name] is None, name
        # the rest of the public surface is unaffected by the missing helpers
        assert ns["Trainer"] is backpropagate.Trainer
        assert ns["safe_path"] is backpropagate.safe_path
        assert isinstance(ns["__version__"], str)

    def test_none_names_are_declared_in___all__(self, monkeypatch):
        ns = _exec_init(monkeypatch, blocked=("backpropagate.ui_security",))
        for name in UI_SECURITY_NAMES:
            assert name in ns["__all__"], name


class TestVersionFallback:
    def test_installed_version_is_reported(self, monkeypatch):
        ns = _exec_init(monkeypatch)
        assert ns["__version__"] == backpropagate.__version__
        assert ns["__version__"] != "0.0.0+unknown"

    def test_source_checkout_without_metadata_gets_the_local_sentinel(self, monkeypatch):
        ns = _exec_init(monkeypatch, version_error=True)
        assert ns["__version__"] == "0.0.0+unknown"
        assert isinstance(ns["__version__"], str)  # callers never hit an AttributeError

    def test_sentinel_is_a_valid_pep440_local_version(self):
        from packaging.version import Version

        v = Version("0.0.0+unknown")
        assert v.local == "unknown" and v.base_version == "0.0.0"


class TestDeprecatedUiAttrs:
    def test_removed_launch_is_an_attribute_error_with_the_cli_hint(self):
        """The v1.8 cut: no DeprecationWarning grace any more, a plain AttributeError."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no DeprecationWarning is emitted
            with pytest.raises(AttributeError, match="backprop ui --port 7862"):
                backpropagate.launch  # noqa: B018

    def test_unknown_attribute_is_a_plain_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute 'definitely_missing'"):
            backpropagate.definitely_missing  # noqa: B018

    def test_exec_namespace_does_not_replace_the_real_module(self, monkeypatch):
        before = sys.modules["backpropagate"]
        _exec_init(monkeypatch, blocked=("backpropagate.ui_security",))
        assert sys.modules["backpropagate"] is before
        assert isinstance(before, types.ModuleType)
        assert backpropagate.SecurityConfig is not None
