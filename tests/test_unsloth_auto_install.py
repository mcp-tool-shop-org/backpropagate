"""backpropagate never lets Unsloth install system software unasked.

Unsloth's GGUF export, left to its default, runs ``winget install`` (Windows)
or apt / brew for CMake, compilers and OpenSSL, accepting their licence
agreements. On 2026-09-30 a plain ``backprop export --format gguf`` installed
CMake and VS Build Tools on the dev rig that way. Unsloth reads
``UNSLOTH_AUTO_INSTALL`` when it attempts the install, and importing
backpropagate must set it to "0" before any Unsloth import, unless the operator
opts in with ``BACKPROPAGATE_UNSLOTH_AUTO_INSTALL``.

These tests put a stub ``unsloth`` package first on ``sys.path`` in a fresh
interpreter. The stub records the value of ``UNSLOTH_AUTO_INSTALL`` at the
moment it is imported. No real Unsloth is involved.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _stub_unsloth(tmp_path: Path) -> Path:
    pkg = tmp_path / "stubs" / "unsloth"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(textwrap.dedent("""
        import os
        with open(os.environ["STUB_RECORD"], "a", encoding="utf-8") as fh:
            fh.write(repr(os.environ.get("UNSLOTH_AUTO_INSTALL")) + "\\n")
    """))
    return pkg.parent


def _run(tmp_path: Path, code: str, **env_overrides: str) -> list[str]:
    record = tmp_path / "record.txt"
    env = {k: v for k, v in os.environ.items()
           if k not in ("UNSLOTH_AUTO_INSTALL", "BACKPROPAGATE_UNSLOTH_AUTO_INSTALL")}
    env["PYTHONPATH"] = os.pathsep.join([str(_stub_unsloth(tmp_path)), str(_REPO_ROOT)])
    env["STUB_RECORD"] = str(record)
    env.update(env_overrides)
    proc = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return record.read_text(encoding="utf-8").split() if record.exists() else []


# Every route that reaches Unsloth: the export probe, the trainer module, and
# the CLI module. Each imports backpropagate first, as users do.
_ENTRY_POINTS = [
    "from backpropagate.export import _has_unsloth; assert _has_unsloth()",
    "import backpropagate.trainer, unsloth",
    "import backpropagate.cli, unsloth",
]


@pytest.mark.parametrize("code", _ENTRY_POINTS)
def test_auto_install_is_off_before_unsloth_is_imported(tmp_path: Path, code: str) -> None:
    assert _run(tmp_path, code) == ["'0'"]


def test_existing_env_value_is_overridden(tmp_path: Path) -> None:
    """A stray UNSLOTH_AUTO_INSTALL=1 in the shell is not an opt-in."""
    assert _run(tmp_path, _ENTRY_POINTS[0], UNSLOTH_AUTO_INSTALL="1") == ["'0'"]


def test_explicit_opt_in_is_honoured(tmp_path: Path) -> None:
    assert _run(
        tmp_path, _ENTRY_POINTS[0], BACKPROPAGATE_UNSLOTH_AUTO_INSTALL="1"
    ) == ["'1'"]
