"""Shared plumbing for the Atheris harnesses in ``fuzz/``.

Every harness is split in two so the same property code runs everywhere:

* ``check_<target>(data: bytes) -> None`` is pure Python. It raises
  ``AssertionError`` when a property is violated and lets any *unexpected*
  exception escape (an exception outside the documented set is a finding).
  ``tests/test_fuzz_harnesses.py`` drives it with the seed corpus on every
  platform, with no Atheris installed.
* ``TestOneInput(data)`` wraps the check for libFuzzer, and ``main()`` hands it
  to ``atheris.Fuzz()``. Atheris ships Linux wheels only, so on Windows the
  import is guarded and ``main()`` prints a clear message instead of failing.

Run one harness (Linux, Python 3.12+; see ``requirements/fuzz.txt``)::

    python fuzz/fuzz_datasets.py -atheris_runs=100000 fuzz/corpus/datasets

Seeds in ``fuzz/corpus/<target>/`` are raw bytes in each harness's own encoding
(see ``Provider``); the ``cov-*.seed`` files are a coverage-minimised
(``-merge=1``) set found by the fuzzer, the named ones are hand-written.

Known findings
--------------
Fuzzing found real bugs. The library is NOT fixed in the change that added this
directory, so each harness steps around them with a ``strict`` switch: with
``strict=False`` (what the fuzzer and the corpus tests use) the known input
class is tolerated so a run can continue and find *new* bugs; with
``strict=True`` (``BP_FUZZ_STRICT=1``, or the ``strict`` input of the fuzz
workflow) the same property is enforced and the reproducer fails. Each has a
strict xfail regression test in ``tests/test_fuzz_harnesses.py``. When one is
fixed, delete its tolerance here and its xfail marker there together.

* F1  ``validate_dataset`` / ``DatasetLoader`` raise ``TypeError`` on an OpenAI
  message whose ``role`` is a JSON list or object.
* F2  ``sanitize_filename`` can return ``"."`` or ``".."``.
* F3  ``sanitize_filename`` can return more than 255 characters.
* F4  ``validate_numeric_input`` accepts NaN inside any min/max range; F4b: so do
  the ``TrainingConfig`` "reject non-positive" validators.
* F5  ``validate_numeric_input`` raises ``OverflowError`` for an int above 1e308.
* F6  ``deduplicate_exact`` raises ``UnicodeEncodeError`` on a lone surrogate.
* F7  ``safe_path`` leaks ``RuntimeError`` / ``OSError`` for a symlink loop or a
  path through a file.
* F8  ``_redact_paths`` leaves the second word of a two-word user name behind.
* F9  ``StreamingDatasetLoader`` leaks ``RecursionError`` on deeply nested JSON
  where ``DatasetLoader`` raises ``ValueError``.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parent.parent
FUZZ_DIR = Path(__file__).resolve().parent

# Fuzz the checked-out source tree, not whatever happens to be installed.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# BP_FUZZ_STRICT=1 turns the known-finding tolerances off, so the fuzzer (or a
# regression test) fails on the bugs listed in the PR that added this directory.
STRICT = os.environ.get("BP_FUZZ_STRICT") == "1"

try:  # Atheris ships Linux wheels only; stay importable everywhere else.
    import atheris
except ImportError:  # pragma: no cover - exercised on Windows / macOS
    atheris = None  # type: ignore[assignment]

T = TypeVar("T")


def instrument(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """Coverage-instrument ``backpropagate.*`` imports, but only when fuzzing.

    ``enabled`` is ``__name__ == "__main__"`` at the call site, so importing a
    harness from pytest never pays the instrumentation cost. Third-party
    packages (pydantic, ...) are left uninstrumented on purpose: the target is
    this repo's parsing code, and instrumenting a validation framework burns
    most of the executions on code we do not own.
    """
    if enabled and atheris is not None:
        return atheris.instrument_imports(include=["backpropagate"])
    return contextlib.nullcontext()


def main(test_one_input: Any) -> int:
    """Entry point shared by every harness (``sys.exit(main(TestOneInput))``)."""
    if atheris is None:
        print(
            "atheris is not installed (it ships Linux wheels only). Install it "
            "with `pip install --require-hashes -r requirements/fuzz.txt` on "
            "Linux, or run the property code without a fuzzer via "
            "`pytest tests/test_fuzz_harnesses.py`.",
            file=sys.stderr,
        )
        return 2
    # The libraries under test log a warning per hostile input; at fuzzing speed
    # that is millions of lines and most of the runtime.
    logging.disable(logging.CRITICAL)

    # libFuzzer's own "Done N runs" line is missing when a run ends on
    # -max_total_time, and -print_final_stats is not honoured by Atheris, so the
    # harness counts its own executions for the workflow summary.
    executions = 0

    def counted(data: bytes) -> None:
        nonlocal executions
        executions += 1
        test_one_input(data)

    atexit.register(lambda: print(f"bp-fuzz: executions={executions}", file=sys.stderr))
    atheris.Setup(sys.argv, counted)
    atheris.Fuzz()
    return 0


class Provider:
    """Tiny, deterministic byte-stream consumer (a subset of FuzzedDataProvider).

    Used instead of ``atheris.FuzzedDataProvider`` so that a corpus file means
    exactly the same thing under Atheris and in the plain-pytest corpus tests.
    Running out of data is not an error: every method degrades to a neutral
    value, which keeps short inputs valid and lets libFuzzer minimise freely.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def remaining(self) -> int:
        return len(self._data) - self._pos

    def bytes(self, n: int) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def rest(self) -> bytes:
        return self.bytes(self.remaining())

    def int_in_range(self, lo: int, hi: int) -> int:
        span = hi - lo + 1
        nbytes = max(1, (span.bit_length() + 7) // 8)
        raw = int.from_bytes(self.bytes(nbytes), "little")
        return lo + raw % span

    def bool(self) -> bool:
        return self.int_in_range(0, 1) == 1

    def pick(self, seq: Sequence[T]) -> T:
        return seq[self.int_in_range(0, len(seq) - 1)]

    def text(self, max_len: int = 40) -> str:
        """A short string; invalid UTF-8 is replaced, lone surrogates survive."""
        n = self.int_in_range(0, max_len)
        return self.bytes(n * 4).decode("utf-8", errors="replace")[:n]

    def tokens(self, vocab: Sequence[str], max_items: int, sep: Sequence[str]) -> str:
        """Join up to ``max_items`` vocabulary words with chosen separators."""
        out: list[str] = []
        for _ in range(self.int_in_range(0, max_items)):
            out.append(self.pick(vocab))
            out.append(self.pick(sep))
        return "".join(out)


def has_unhashable_role(rows: Any) -> bool:
    """True if any OpenAI-style message carries an unhashable ``role`` value.

    Known finding F1: ``_validate_openai`` tests ``msg["role"] not in
    valid_roles`` (a ``set``), which raises ``TypeError`` for a JSON list or
    object. See ``tests/test_fuzz_harnesses.py::TestKnownFindings``.
    """
    if not isinstance(rows, list):
        rows = [rows]
    for row in rows:
        if not isinstance(row, dict):
            continue
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            continue
        for msg in msgs:
            if isinstance(msg, dict) and isinstance(msg.get("role"), (list, dict)):
                return True
    return False


_SCRATCH: Path | None = None


def scratch_dir() -> Path:
    """One temp directory per process, removed at exit (not per execution)."""
    global _SCRATCH
    if _SCRATCH is None:
        import atexit
        import shutil
        import tempfile

        _SCRATCH = Path(tempfile.mkdtemp(prefix="bp-fuzz-")).resolve()
        atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)
    return _SCRATCH


@contextlib.contextmanager
def patched_environ(updates: dict[str, str]) -> Iterator[None]:
    """Temporarily overlay ``os.environ`` (restored even if the check raises)."""
    saved = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
