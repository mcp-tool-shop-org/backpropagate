#!/usr/bin/env python3
"""Atheris harness: path validation and the UI path sandbox.

Targets:

* ``backpropagate.security.safe_path`` -- the library's path-traversal guard,
  with and without ``allowed_base``;
* ``backpropagate.ui_security._is_forbidden_output_base`` -- the denylist behind
  the UI output sandbox (system trees, credential directories, symlinks under
  ``$HOME``);
* ``backpropagate.ui_security.sanitize_filename`` -- applied to every uploaded
  file name.

Candidate paths are built from a vocabulary of the strings that matter
(``..``, ``.``, symlinks that point out of the sandbox, symlink loops, NUL,
backslashes, look-alike dots) rather than raw bytes, so the fuzzer spends its
time on structure. A real directory tree is created once per process::

    <scratch>/base/{sub/deep, file.txt, link_out -> ../outside, link_in -> sub,
                    loop_a <-> loop_b}
    <scratch>/outside/secret.txt
    <scratch>/home/{.ssh, .config, work/link_ssh -> ../.ssh}      (fake $HOME)

Properties:

* a path ``safe_path(..., allowed_base=base)`` accepts never resolves outside
  ``base`` (checked against an independent ``os.path.realpath`` oracle), and on
  POSIX the accept/reject decision matches that oracle exactly;
* without a base, a ``..`` path is accepted only if it is relative and lands
  inside the working directory;
* only documented exceptions escape (``PathTraversalError``, ``ValueError`` for
  NUL / ``allow_relative=False``, ``FileNotFoundError`` for ``must_exist``);
* ``_is_forbidden_output_base`` never says "allowed" for something that really
  resolves into a system tree or a credential directory;
* ``sanitize_filename`` returns a non-empty, separator-free, control-free name of
  at most 255 characters that is not ``.`` or ``..``.

Run: ``python fuzz/fuzz_paths.py -atheris_runs=200000 fuzz/corpus/paths``
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from fuzz_common import STRICT, Provider, instrument, main, patched_environ, scratch_dir

with instrument(__name__ == "__main__"):
    from backpropagate.security import PathTraversalError, safe_path
    from backpropagate.ui_security import _is_forbidden_output_base, sanitize_filename

PATH_VOCAB = (
    "sub", "deep", "file.txt", "link_out", "link_in", "loop_a", "loop_b",
    "outside", "secret.txt", "..", "..", ".", "", "~", "%2e%2e", "..\\", "\x00",
    chr(0x2024) * 2, chr(0xFF0E) * 2, "base", "con", " ", "...", "a" * 40, chr(0x202E), "x",
)  # fmt: skip
PATH_SEPS = ("/", "/", "//", "\\", "/./", "")

FORBIDDEN_VOCAB = (
    ".ssh", ".config", ".aws", "..", "..", ".", "work", "link_ssh", "a", "etc",
    "passwd", "bin", "lib", "run", "ui-outputs", "proc", "self", "root",
)  # fmt: skip

POSIX = os.name == "posix"
_TREE: dict[str, Path] = {}


def tree() -> dict[str, Path]:
    """Build (once) the sandbox described in the module docstring."""
    if _TREE:
        return _TREE
    scratch = scratch_dir()
    base, outside, home = scratch / "base", scratch / "outside", scratch / "home"
    for d in (base / "sub" / "deep", outside, home / ".ssh", home / ".config", home / "work"):
        d.mkdir(parents=True, exist_ok=True)
    (base / "file.txt").write_text("x")
    (outside / "secret.txt").write_text("secret")
    # Symlinks need privileges on Windows; without them those paths simply
    # never exist and the remaining properties still hold.
    for target, link in (
        ("../outside", base / "link_out"),
        ("sub", base / "link_in"),
        ("loop_b", base / "loop_a"),
        ("loop_a", base / "loop_b"),
        ("../.ssh", home / "work" / "link_ssh"),
    ):
        with contextlib.suppress(OSError, NotImplementedError):
            os.symlink(target, link)
    _TREE.update(scratch=scratch, base=base, outside=outside, home=home)
    return _TREE


def _inside(path: str | Path, root: str | Path) -> bool:
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(root))


def _has_control(name: str) -> bool:
    return any(ord(c) < 0x20 or 0x7F <= ord(c) <= 0x9F for c in name)


# --------------------------------------------------------------------------
# safe_path
# --------------------------------------------------------------------------


def _candidate(p: Provider, anchors: tuple[str, ...]) -> str:
    return p.pick(anchors) + p.tokens(PATH_VOCAB, 8, PATH_SEPS) + (p.text(6) if p.bool() else "")


def check_safe_path_with_base(p: Provider, strict: bool = False) -> None:
    t = tree()
    base = t["base"]
    anchors = (f"{base}/", str(base), f"{base}/sub/", "", "/", f"{t['scratch']}/", f"{t['outside']}/")
    cand = _candidate(p, anchors)

    try:
        out = safe_path(cand, allowed_base=base)
    except PathTraversalError:
        accepted = False
    except ValueError:
        assert "\x00" in cand, f"unexpected ValueError for {cand!r}"
        return
    except (OSError, RuntimeError) as exc:
        # Known finding F7: resolution errors (a path through a file, a symlink
        # loop) escape as a raw OSError / RuntimeError on some platforms.
        assert not strict, f"safe_path leaked {exc!r} for {cand!r}"
        return
    else:
        accepted = True
        assert out.is_absolute() or not POSIX
        assert ".." not in out.parts
        assert _inside(out, base), f"{cand!r} was accepted but resolves to {out}"

    if POSIX:
        # Exact differential oracle: accepted <=> realpath lands inside base.
        assert accepted == _inside(cand, base), (
            f"{cand!r}: safe_path accepted={accepted} but oracle says "
            f"inside={_inside(cand, base)}"
        )


def check_safe_path_no_base(p: Provider, strict: bool = False) -> None:
    t = tree()
    anchors = (f"{t['base']}/", "", "/", "./", "../", f"{t['outside']}/")
    cand = _candidate(p, anchors)
    relative_ok = p.bool()
    must_exist = p.bool()
    cwd = Path.cwd()

    try:
        out = safe_path(cand, must_exist=must_exist, allow_relative=relative_ok)
    except PathTraversalError:
        assert ".." in cand, f"{cand!r} rejected as traversal without any '..'"
        return
    except FileNotFoundError:
        assert must_exist
        return
    except (OSError, RuntimeError) as exc:
        # Known finding F7 (see check_safe_path_with_base).
        assert not strict, f"safe_path leaked {exc!r} for {cand!r}"
        return
    except ValueError as exc:
        assert "\x00" in cand or (not relative_ok and not Path(cand).is_absolute()), (
            f"unexpected ValueError for {cand!r}: {exc}"
        )
        return

    assert out.is_absolute() or not POSIX
    if must_exist:
        assert out.exists()
    if not relative_ok:
        assert Path(cand).is_absolute()
    if ".." in cand:
        assert not Path(cand).is_absolute(), f"absolute '..' path accepted: {cand!r}"
        assert _inside(out, cwd), f"'..' path escaped the working directory: {cand!r} -> {out}"


# --------------------------------------------------------------------------
# sanitize_filename
# --------------------------------------------------------------------------

NAME_VOCAB = (
    "..", ".", " ", "\x01", "\x85", "\x7f", "a", "b.txt", "/", "\\", "\x00", ".jsonl",
    chr(0xE9), "x" * 150, "x" * 300, ".", "..",
)  # fmt: skip


def check_sanitized(name: str) -> None:
    out = sanitize_filename(name)

    assert out, "empty file name"
    assert "/" not in out and "\\" not in out and chr(0) not in out
    assert not _has_control(out)

    assert out not in (".", ".."), f"sanitize_filename({name!r}) returned {out!r}"
    assert len(out) <= 255, f"sanitize_filename returned {len(out)} chars"


def check_sanitize_filename(p: Provider) -> None:
    name = p.tokens(NAME_VOCAB, 8, ("", "", ".")) + (p.text(8) if p.bool() else "")
    check_sanitized(name)


# --------------------------------------------------------------------------
# _is_forbidden_output_base
# --------------------------------------------------------------------------

SYSTEM_TREES = ("/etc", "/proc", "/sys", "/dev", "/boot", "/var/run", "/var/lib",
                "/usr", "/bin", "/sbin", "/root")  # fmt: skip
CREDENTIAL_DIRS = (".ssh", ".aws", ".kube", ".docker", ".gnupg", ".config")


def check_forbidden_output_base(p: Provider) -> None:
    t = tree()
    home = t["home"]
    roots = (str(home), f"{home}/work", f"{home}/.ssh", f"{home}/.config", str(t["scratch"]),
             "/etc", "/usr", "/proc", "/", "", "/var/lib", "/tmp", "/dev/shm")  # fmt: skip
    cand = p.pick(roots) + "/" + p.tokens(FORBIDDEN_VOCAB, 8, ("/", "/", "//"))
    cand = cand.replace("\x00", "")

    with patched_environ({"HOME": str(home), "USERPROFILE": str(home), "APPDATA": ""}):
        forbidden = _is_forbidden_output_base(Path(cand))
        if forbidden:
            return  # fail-closed is always acceptable

        real = os.path.realpath(cand)
        home_real = os.path.realpath(home)
        if Path(real).is_relative_to(home_real):
            for sub in CREDENTIAL_DIRS:
                cred = os.path.realpath(Path(home) / sub)
                assert not Path(real).is_relative_to(cred), (
                    f"{cand!r} resolves into credential dir {cred} but was allowed"
                )
        elif POSIX:
            for tree_root in SYSTEM_TREES:
                sys_real = os.path.realpath(tree_root)
                assert not Path(real).is_relative_to(sys_real), (
                    f"{cand!r} resolves into {sys_real} but was allowed"
                )


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def check_paths(data: bytes, strict: bool = False) -> None:
    p = Provider(data)
    mode = p.int_in_range(0, 3)
    if mode == 0:
        check_safe_path_with_base(p, strict=strict)
    elif mode == 1:
        check_safe_path_no_base(p, strict=strict)
    elif mode == 2:
        check_sanitize_filename(p)
    else:
        check_forbidden_output_base(p)


def TestOneInput(data: bytes) -> None:
    check_paths(data, strict=STRICT)


if __name__ == "__main__":
    sys.exit(main(TestOneInput))
