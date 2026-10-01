#!/usr/bin/env python3
"""Atheris harness: UI input validation and output sanitisation.

Target: the string / number / file-content gatekeepers in
``backpropagate/ui_security.py`` that sit between a (possibly remote,
``--share --auth``) UI user and the trainer:

* ``validate_string_input`` / ``validate_numeric_input`` -- field validation;
* ``validate_auth_shape`` -- the launch-time ``auth=`` shape check;
* ``validate_file_magic`` -- magic-byte / HTML-in-disguise upload sniffing;
* ``safe_markdown_fence`` -- wraps untrusted text for display;
* ``_redact_paths`` / ``sanitize_error_for_user`` -- keep paths and internals out
  of error toasts.

Properties asserted (beyond "only documented exceptions escape"):

* a validator that returns a value returns one inside the bounds it was given;
* ``validate_auth_shape`` accepts exactly: ``None``, a callable, a
  ``(non-empty str, non-empty str)`` tuple, or a non-empty list of those tuples,
  and otherwise raises ``BackpropagateError(code="INPUT_AUTH_INVALID_SHAPE")``;
* ``validate_file_magic`` never raises, and never accepts a header that starts
  with a known HTML / script / executable signature;
* ``safe_markdown_fence`` always uses a fence strictly longer than any backtick
  run in the content, so the content cannot close its own fence;
* ``_redact_paths`` leaves none of a user's name behind and is idempotent;
  ``sanitize_error_for_user`` honours its length cap and never echoes the text of
  a non-structured exception.

Run: ``python fuzz/fuzz_ui_input.py -atheris_runs=200000 fuzz/corpus/ui_input``
"""

from __future__ import annotations

import math
import re
import sys
from typing import Any

from fuzz_common import STRICT, Provider, instrument, main, scratch_dir

with instrument(__name__ == "__main__"):
    from backpropagate.exceptions import BackpropagateError, UserInputError
    from backpropagate.ui_security import (
        _PATH_REDACTION_PATTERNS,
        FILE_SIGNATURES,
        _redact_paths,
        safe_markdown_fence,
        sanitize_error_for_user,
        validate_auth_shape,
        validate_file_magic,
        validate_numeric_input,
        validate_string_input,
    )

# --------------------------------------------------------------------------
# validate_string_input
# --------------------------------------------------------------------------

STRING_PATTERNS = (None, r"^[A-Za-z0-9_.-]+$", r"^[\w.-]+/[\w.-]+$", r"\d+")


def check_string_input(p: Provider) -> None:
    value: Any = p.pick((p.text(30), p.text(30), None, 7, 2.5, True, b"bytes"))
    max_length = p.pick((0, 1, 5, 40, 1000))
    min_length = p.pick((0, 1, 3, 10))
    pattern = p.pick(STRING_PATTERNS)
    allow_none, allow_empty = p.bool(), p.bool()

    try:
        out = validate_string_input(
            value,
            "field",
            max_length=max_length,
            min_length=min_length,
            pattern=pattern,
            allow_none=allow_none,
            allow_empty=allow_empty,
        )
    except UserInputError:
        return
    if out is None:
        assert value is None and allow_none
        return
    assert isinstance(out, str)
    assert "\x00" not in out
    assert len(out) <= max_length
    assert len(out) >= min_length
    assert allow_empty or out.strip()
    if pattern is not None:
        assert re.match(pattern, out)


# --------------------------------------------------------------------------
# validate_numeric_input
# --------------------------------------------------------------------------

NUMERIC_TEXT = ("0", "1", "-1", "0.5", "1e3", "1e308", "1e999", "-1e999", "nan", "NaN", "inf",
                "-inf", "1_0", " 5 ", "0x10", "\u0663", "", "1" * 400)  # fmt: skip


def check_numeric(value: Any, lo: float | None, hi: float | None, strict: bool = False) -> None:
    try:
        out = validate_numeric_input(value, "n", min_value=lo, max_value=hi, allow_none=False)
    except UserInputError:
        return
    except OverflowError:
        # Known finding F5: ``float(10**400)`` raises OverflowError, which the
        # validator does not translate into a UserInputError.
        assert not strict, f"OverflowError escaped for {value!r}"
        assert isinstance(value, int) and abs(value) > 10**308
        return

    assert isinstance(out, float)
    if math.isnan(out):
        # Known finding F4: NaN compares False against both bounds, so it
        # slips through any min/max range.
        assert not strict, f"NaN accepted for {value!r} with bounds ({lo}, {hi})"
        return
    if lo is not None:
        assert out >= lo
    if hi is not None:
        assert out <= hi


def check_numeric_input(p: Provider, strict: bool = False) -> None:
    kind = p.int_in_range(0, 3)
    if kind == 0:
        value: Any = p.pick(NUMERIC_TEXT)
    elif kind == 1:
        value = p.int_in_range(-(2**40), 2**40) * 10 ** p.int_in_range(0, 400)
    elif kind == 2:
        value = p.int_in_range(-1000, 1000) / max(p.int_in_range(1, 50), 1)
    else:
        value = p.pick((None, [], {}, object(), True, b"1"))
    lo = p.pick((None, 0.0, -1.0, 1e-9))
    hi = p.pick((None, 1.0, 100.0, 1e308))
    check_numeric(value, lo, hi, strict=strict)


# --------------------------------------------------------------------------
# validate_auth_shape
# --------------------------------------------------------------------------


def _pair_ok(v: Any) -> bool:
    return (
        isinstance(v, tuple)
        and len(v) == 2
        and all(isinstance(x, str) and x for x in v)
    )


def _auth_value(p: Provider, depth: int = 0) -> Any:
    atoms = (None, "", "user", "p", 0, 1.5, True, b"u")
    kind = p.int_in_range(0, 5 if depth < 2 else 1)
    if kind <= 1:
        return p.pick(atoms)
    if kind == 2:
        return tuple(_auth_value(p, depth + 1) for _ in range(p.int_in_range(0, 3)))
    if kind == 3:
        return [_auth_value(p, depth + 1) for _ in range(p.int_in_range(0, 3))]
    if kind == 4:
        return {"user": "pass"}
    return ("user", "pass")


def check_auth_shape(p: Provider) -> None:
    auth = _auth_value(p)
    expected_ok = (
        auth is None
        or _pair_ok(auth)
        or (isinstance(auth, list) and bool(auth) and all(_pair_ok(a) for a in auth))
    )
    try:
        validate_auth_shape(auth)
    except BackpropagateError as exc:
        assert exc.code == "INPUT_AUTH_INVALID_SHAPE"
        assert not expected_ok, f"valid auth rejected: {auth!r}"
    else:
        assert expected_ok, f"invalid auth accepted: {auth!r}"


# --------------------------------------------------------------------------
# validate_file_magic
# --------------------------------------------------------------------------

SUSPICIOUS = (
    b"<!DOCTYPE", b"<html", b"<HTML", b"<script", b"<SCRIPT", b"<?php", b"<?PHP", b"#!/",
    b"MZ", b"\x7fELF", b"\xca\xfe\xba\xbe", b"PK\x03\x04",
)  # fmt: skip
EXTENSIONS = (*FILE_SIGNATURES, ".html", ".bin", "", ".JSONL")


def check_file_magic(p: Provider) -> None:
    ext = p.pick(EXTENSIONS)
    body = p.bytes(p.int_in_range(0, 32))
    if p.bool():
        body = p.pick(SUSPICIOUS) + body
    path = scratch_dir() / "magic.bin"
    path.write_bytes(body)

    ok, message = validate_file_magic(path, expected_extension=ext)
    assert isinstance(ok, bool) and isinstance(message, str)
    header = body[:16]
    if header.startswith(SUSPICIOUS):
        assert not ok, f"{header!r} accepted as {ext!r}"
    if ok and FILE_SIGNATURES.get(ext):
        assert header.startswith(tuple(FILE_SIGNATURES[ext]))


# --------------------------------------------------------------------------
# safe_markdown_fence
# --------------------------------------------------------------------------


def check_markdown_fence(p: Provider) -> None:
    pieces = [p.text(12), "`", "``", "```", "````", "\n", "\r", "  ", "~~~"]
    content = "".join(p.pick(pieces) for _ in range(p.int_in_range(0, 12)))
    language = p.pick(("", "json", "text", "chatml"))

    out = safe_markdown_fence(content, language)
    first, _, rest = out.partition("\n")
    fence = re.match(r"`+", first).group(0)  # type: ignore[union-attr]
    longest = max((len(m) for m in re.findall(r"`+", content)), default=0)
    assert len(fence) >= 3 and len(fence) > longest, "fence not longer than content run"
    assert first == fence + language
    assert rest == content + "\n" + fence, "content was altered or the fence is not closed"
    # No content line may look like a closing fence (>= fence length of backticks).
    for line in content.split("\n"):
        m = re.fullmatch(r" {0,3}(`+)[ \t]*", line)
        assert not (m and len(m.group(1)) >= len(fence))


# --------------------------------------------------------------------------
# Path redaction / user-facing error text
# --------------------------------------------------------------------------

ROOTS = ("/home/", "/Users/", "C:\\Users\\", "D:\\Users\\", "\\\\srv\\share\\", "/tmp/",
         "~/.cache/huggingface/", "/root/")  # fmt: skip
SEPS = ("/", "\\")
PROSE = ("could not open ", "error: ", " at ", "\n", " ", ": ", "'", '"')


def check_redaction_leak(root: str, user: str, sep: str, tail: str = "data.jsonl") -> None:
    """Text built around ``root + user + sep + tail`` must not keep ``user``."""
    text = f"error: cannot read {root}{user}{sep}{tail} (permission denied)"
    out = _redact_paths(text)
    assert user not in out, f"user name leaked: {out!r}"
    assert _redact_paths(out) == out, "redaction is not idempotent"


def check_redaction(p: Provider) -> None:
    root = p.pick(ROOTS)
    user = "u" + p.pick(("alice", "Bob", "x9", "j.doe", "carol_q")) + str(p.int_in_range(0, 999))
    check_redaction_leak(root, user, p.pick(SEPS))

    text = "".join(p.pick(ROOTS + PROSE) + p.text(4) for _ in range(p.int_in_range(0, 6)))
    once = _redact_paths(text)
    for pattern in _PATH_REDACTION_PATTERNS:
        assert not pattern.search(once), f"{pattern.pattern!r} still matches {once!r}"


def check_error_sanitizing(p: Provider) -> None:
    max_length = p.int_in_range(1, 120)
    body = p.text(60) + p.pick(ROOTS) + p.text(10)
    if p.bool():
        exc: BaseException = BackpropagateError(
            body,
            suggestion=body if p.bool() else None,
            code="INPUT_VALIDATION_FAILED",
        )
        message, suggestion = sanitize_error_for_user(exc, "op", max_length=max_length)
        assert message.startswith("INPUT_VALIDATION_FAILED: ")
        assert len(message) - len("INPUT_VALIDATION_FAILED: ") <= max_length
        if suggestion is not None:
            assert len(suggestion) <= max_length
        for pattern in _PATH_REDACTION_PATTERNS:
            assert not pattern.search(message)
    else:
        marker = "ZZsecretZZ" + body
        message, suggestion = sanitize_error_for_user(ValueError(marker), "uploading")
        assert suggestion is None
        assert "ZZsecretZZ" not in message, "raw exception text reached the user"
        assert "uploading" in message


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def check_ui_input(data: bytes, strict: bool = False) -> None:
    p = Provider(data)
    mode = p.int_in_range(0, 6)
    if mode == 0:
        check_string_input(p)
    elif mode == 1:
        check_numeric_input(p, strict=strict)
    elif mode == 2:
        check_auth_shape(p)
    elif mode == 3:
        check_file_magic(p)
    elif mode == 4:
        check_markdown_fence(p)
    elif mode == 5:
        check_redaction(p)
    else:
        check_error_sanitizing(p)


def TestOneInput(data: bytes) -> None:
    check_ui_input(data, strict=STRICT)


if __name__ == "__main__":
    sys.exit(main(TestOneInput))
