"""Contract tests for backpropagate/ui_marker_warmup.py.

The warmup replaces a PRIVATE reflex function (``js_runtimes._install_frontend_packages``)
and re-computes its cache fingerprint with another private one. Reflex moved that signature
between 0.9.5 and 0.9.12 (the hook then wrote a marker for the wrong arguments, silently), so
these tests fail loudly the next time a Reflex bump moves it again.
"""

from __future__ import annotations

import inspect
import pickle
import runpy
import types

import pytest

from backpropagate import ui_marker_warmup as warm

js_runtimes = pytest.importorskip("reflex.utils.js_runtimes")


def test_reflex_install_signature_is_the_one_the_warmup_mirrors():
    # `_install_frontend_packages` is `cached_procedure(payload_fn=_frontend_packages_cache_payload)`:
    # the wrapper takes the payload function's arguments, so that is where the shape is readable.
    fn = js_runtimes._frontend_packages_cache_payload
    assert tuple(inspect.signature(fn).parameters) == warm.EXPECTED_INSTALL_PARAMS, (
        "reflex.utils.js_runtimes._frontend_packages_cache_payload changed shape; update "
        "ui_marker_warmup.warm and EXPECTED_INSTALL_PARAMS together"
    )
    assert warm.install_signature_matches(fn)


def test_a_drifted_signature_is_refused_not_guessed():
    def old_shape(packages, config, install_package_managers):  # reflex 0.9.5
        return ""

    assert not warm.install_signature_matches(old_shape)
    assert not warm.install_signature_matches(None)


def test_main_exits_3_when_reflex_internals_moved(monkeypatch):
    fake = types.SimpleNamespace(
        _frontend_packages_cache_payload=lambda a, b, c: "",
        _frontend_packages_cache_path=lambda: None,
    )
    monkeypatch.setattr(js_runtimes, "_frontend_packages_cache_payload", fake._frontend_packages_cache_payload)
    assert warm.main(["x", "3000", "127.0.0.1"]) == 3


def test_marker_is_reflexs_own_fingerprint_for_the_arguments_it_was_called_with(
    monkeypatch, tmp_path
):
    marker = tmp_path / "reflex.install_frontend_packages.cached"
    monkeypatch.setattr(js_runtimes, "_frontend_packages_cache_path", lambda: marker)
    # Restore the real function afterwards: main() replaces it on the module.
    monkeypatch.setattr(
        js_runtimes, "_install_frontend_packages", js_runtimes._install_frontend_packages
    )
    args = ({"left-pad@1.0.0"}, {"typescript@5"}, True, ("/usr/bin/bun",))

    def fake_reflex_run(name, run_name=None):
        js_runtimes._install_frontend_packages(*args)
        raise AssertionError("the warm hook must abort the run")

    monkeypatch.setattr(runpy, "run_module", fake_reflex_run)
    assert warm.main(["x", "3000", "127.0.0.1"]) == 0
    payload, value = pickle.loads(marker.read_bytes())  # noqa: S301 - a file this test just wrote
    assert payload == js_runtimes._frontend_packages_cache_payload(*args)
    assert value is None
