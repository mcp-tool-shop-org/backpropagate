#!/usr/bin/env python3
"""Atheris harness: configuration parsed from environment-style strings.

Targets:

* the pydantic-settings classes in ``backpropagate/config.py`` (``ModelConfig``,
  ``LoRAConfig``, ``TrainingConfig``, ``DataConfig``, ``UIConfig``,
  ``MultiRunSettings``, ``SecurityConfig``), populated through
  ``BACKPROPAGATE_<SECTION>__<FIELD>`` environment variables;
* ``ui_security.load_config_from_env``, the hand-rolled ``int`` / ``bool``
  parser behind the UI security limits.

Configuration reaches the library as untrusted strings (shell profiles, ``.env``
files, container env, CI secrets), so a malformed value must fail *loudly and
structurally*, and a value that is accepted must honour the invariants the
validators document.

Properties:

* construction either succeeds or raises one of pydantic's ``ValidationError``,
  pydantic-settings' ``SettingsError`` or the library's ``BackpropagateError``
  (``InvalidSettingError``); any other exception is a finding;
* construction is deterministic for a given environment;
* every field that is declared as a plain ``int`` / ``float`` / ``bool`` /
  ``str`` holds exactly that type;
* an accepted ``TrainingConfig`` satisfies its documented guards: ``method`` and
  ``backend`` in their allowed sets, not both ``bf16`` and ``fp16``, and
  ``orpo_beta``, ``simpo_gamma`` and both KTO weights strictly positive;
* ``load_config_from_env`` never raises, leaves unparseable values at their
  default, and sets int fields to ``int(value)`` and bool fields to the
  documented truthy-word test.

Run: ``python fuzz/fuzz_config.py -atheris_runs=100000 fuzz/corpus/config``
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

from fuzz_common import STRICT, Provider, instrument, main, patched_environ

with instrument(__name__ == "__main__"):
    from pydantic import ValidationError
    from pydantic_settings.exceptions import SettingsError

    from backpropagate import config as cfg
    from backpropagate.exceptions import BackpropagateError
    from backpropagate.ui_security import SecurityConfig as UiSecurityConfig
    from backpropagate.ui_security import load_config_from_env

ALLOWED = (ValidationError, SettingsError, BackpropagateError)

SECTIONS = {
    "ModelConfig": "BACKPROPAGATE_MODEL__",
    "LoRAConfig": "BACKPROPAGATE_LORA__",
    "TrainingConfig": "BACKPROPAGATE_TRAINING__",
    "DataConfig": "BACKPROPAGATE_DATA__",
    "UIConfig": "BACKPROPAGATE_UI__",
    "MultiRunSettings": "BACKPROPAGATE_MULTIRUN__",
    "SecurityConfig": "BACKPROPAGATE_SECURITY__",
}

VALUES = (
    "true", "false", "True", "1", "0", "-1", "2", "nan", "NaN", "inf", "-inf", "1e400", "1_000",
    "1e3", "0.5", "-0.5", "", " ", "null", "[]", "{}", '{"a": 1}', '["x"]', "sft", "orpo", "simpo",
    "kto", "dpo", "auto", "cuda", "mlx", "rocm", "yes", "off", chr(0x663), "0x10", "9" * 30,
)  # fmt: skip


def _field_names(cls: Any) -> list[str]:
    return list(cls.model_fields)


def _plain_type_ok(cls: Any, name: str, value: Any) -> None:
    ann = cls.model_fields[name].annotation
    if ann in (int, float, bool, str):
        assert type(value) is ann, f"{cls.__name__}.{name}: {value!r} is not {ann.__name__}"


def check_training_invariants(c: Any, strict: bool = False) -> None:
    assert c.method in cfg._ALLOWED_METHODS
    assert c.backend in ("auto", "cuda", "mlx")
    assert not (c.bf16 and c.fp16)
    for name in ("orpo_beta", "simpo_gamma", "kto_desirable_weight", "kto_undesirable_weight"):
        value = getattr(c, name)
        if math.isnan(value):
            # Known finding F4b: ``nan <= 0`` is False, so the "reject
            # non-positive" validators let NaN through.
            assert not strict, f"TrainingConfig.{name} accepted NaN"
            continue
        assert value > 0, f"TrainingConfig.{name}={value!r} passed validation"


def check_section(p: Provider, strict: bool = False) -> None:
    class_name = p.pick(tuple(SECTIONS))
    cls = getattr(cfg, class_name)
    prefix = SECTIONS[class_name]
    names = _field_names(cls)

    env: dict[str, str] = {}
    for _ in range(p.int_in_range(1, 5)):
        field = p.pick(names)
        value = p.pick(VALUES) if p.bool() else p.text(10)
        env[prefix + field.upper()] = value.replace("\x00", "")

    with patched_environ(env):
        try:
            first = cls(_env_file=None)
        except ALLOWED:
            try:
                cls(_env_file=None)
            except ALLOWED:
                return
            raise AssertionError(f"{class_name} rejected {env!r} once and accepted it once")
        second = cls(_env_file=None)

    assert repr(first) == repr(second), "construction is not deterministic"
    for name in names:
        _plain_type_ok(cls, name, getattr(first, name))
    if class_name == "TrainingConfig":
        check_training_invariants(first, strict=strict)
    if class_name == "SecurityConfig":
        assert isinstance(first.validate_production_config(), list)
        auth = first.get_auth_tuple()
        assert auth is None or (auth[0] and auth[1])


def check_ui_security_env(p: Provider) -> None:
    prefix = "BACKPROPAGATE_SECURITY__"
    defaults = UiSecurityConfig()
    int_fields = [n for n, v in vars(defaults).items() if type(v) is int]
    bool_fields = [n for n, v in vars(defaults).items() if type(v) is bool]
    names = int_fields + bool_fields + ["not_a_field"]

    env: dict[str, str] = {}
    for _ in range(p.int_in_range(1, 5)):
        value = p.pick(VALUES) if p.bool() else p.text(10)
        env[prefix + p.pick(names).upper()] = value.replace("\x00", "")

    with patched_environ(env):
        loaded = load_config_from_env(UiSecurityConfig())
        live = {k[len(prefix) :].lower(): v for k, v in os.environ.items() if k.startswith(prefix)}

    for name in int_fields:
        value = getattr(loaded, name)
        assert type(value) is int
        if name in live:
            try:
                expected = int(live[name])
            except ValueError:
                expected = getattr(defaults, name)
            assert value == expected, f"{name}: {live[name]!r} -> {value!r}"
        else:
            assert value == getattr(defaults, name)
    for name in bool_fields:
        value = getattr(loaded, name)
        assert type(value) is bool
        if name in live:
            assert value == (live[name].lower() in ("true", "1", "yes", "on"))
        else:
            assert value == getattr(defaults, name)


def check_config(data: bytes, strict: bool = False) -> None:
    p = Provider(data)
    if p.int_in_range(0, 3) == 0:
        check_ui_security_env(p)
    else:
        check_section(p, strict=strict)


def TestOneInput(data: bytes) -> None:
    check_config(data, strict=STRICT)


if __name__ == "__main__":
    sys.exit(main(TestOneInput))
