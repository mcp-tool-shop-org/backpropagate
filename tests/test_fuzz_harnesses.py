"""CPU tests for the Atheris harnesses in ``fuzz/``.

Atheris ships Linux wheels only, so these tests never import it. Each harness
keeps its property code in a pure-Python ``check_*`` function; here that code is
driven directly with the seed corpus (and a fixed-seed stream of random inputs),
so CI exercises every property on every platform on every run. The real fuzzing
(``.github/workflows/fuzz.yml``, manual dispatch) feeds the very same functions.

The tests are in four groups:

* ``TestCorpus`` -- every seed corpus file and a batch of random inputs pass
  every property;
* ``TestHarnessesHaveTeeth`` -- sabotage the library function a harness guards
  and check that the harness notices (a property check that cannot fail is not a
  check);
* ``TestKnownFindings`` -- fuzzing found real bugs that this PR deliberately does
  not fix. Each is pinned by a ``strict`` xfail regression test (so the day it
  is fixed the test XPASSes loudly and the marker must be removed) and by a seed
  in the corpus that the harness tolerates by default and rejects under
  ``strict=True``;
* ``TestPlumbing`` -- the import guard, the byte provider, and the workflow and
  requirements-file contracts (manual dispatch only, SHA-pinned actions,
  hash-pinned install).
"""

from __future__ import annotations

import importlib
import os
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FUZZ_DIR = ROOT / "fuzz"
CORPUS = FUZZ_DIR / "corpus"

if not CORPUS.is_dir():  # an sdist ships tests/ but not fuzz/
    pytest.skip("fuzz/ is not part of this distribution", allow_module_level=True)

if str(FUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(FUZZ_DIR))

# corpus directory -> (harness module, check function)
TARGETS = {
    "datasets": ("fuzz_datasets", "check_datasets"),
    "dataset_files": ("fuzz_dataset_files", "check_dataset_files"),
    "paths": ("fuzz_paths", "check_paths"),
    "ui_input": ("fuzz_ui_input", "check_ui_input"),
    "config": ("fuzz_config", "check_config"),
}


def load(target: str):
    """Import a harness module and return ``(module, check_function)``."""
    if target == "config":
        pytest.importorskip("pydantic_settings")
    module_name, check_name = TARGETS[target]
    module = importlib.import_module(module_name)
    return module, getattr(module, check_name)


def corpus_files(target: str) -> list[Path]:
    return sorted(p for p in (CORPUS / target).iterdir() if p.is_file())


CORPUS_PARAMS = [
    pytest.param(target, path, id=f"{target}/{path.name}")
    for target in TARGETS
    for path in corpus_files(target)
]


def random_inputs(seed: int, count: int, max_len: int = 96):
    rng = random.Random(seed)
    for _ in range(count):
        yield bytes(rng.randrange(256) for _ in range(rng.randrange(1, max_len)))


def first_failure(check, inputs) -> AssertionError | None:
    """Run ``check`` over ``inputs``; return the first AssertionError (or None)."""
    for data in inputs:
        try:
            check(data)
        except AssertionError as exc:
            return exc
    return None


def corpus_and_random(target: str, count: int = 1500):
    for path in corpus_files(target):
        yield path.read_bytes()
    yield from random_inputs(seed=1234, count=count)


class TestCorpus:
    @pytest.mark.parametrize("target", TARGETS)
    def test_every_target_has_a_seed_corpus(self, target):
        assert len(corpus_files(target)) >= 5

    @pytest.mark.parametrize(("target", "path"), CORPUS_PARAMS)
    def test_seed_passes_every_property(self, target, path):
        _, check = load(target)
        check(path.read_bytes())

    @pytest.mark.parametrize("target", TARGETS)
    def test_random_inputs_pass_every_property(self, target):
        _, check = load(target)
        for data in random_inputs(seed=20260930, count=250):
            check(data)


class TestHarnessesHaveTeeth:
    """Each sabotage reintroduces the bug class a harness exists to catch."""

    def test_paths_catches_a_sandbox_that_ignores_allowed_base(self, monkeypatch):
        module, check = load("paths")
        monkeypatch.setattr(module, "safe_path", lambda cand, **_kw: Path(cand).resolve())
        assert first_failure(check, corpus_and_random("paths", 200)) is not None

    def test_paths_catches_an_empty_forbidden_denylist(self, monkeypatch):
        module, check = load("paths")
        monkeypatch.setattr(module, "_is_forbidden_output_base", lambda _p: False)
        assert first_failure(check, corpus_and_random("paths", 200)) is not None

    def test_paths_catches_a_sanitizer_that_keeps_separators(self, monkeypatch):
        module, check = load("paths")
        monkeypatch.setattr(module, "sanitize_filename", lambda name: name or "x")
        assert first_failure(check, corpus_and_random("paths", 200)) is not None

    def test_ui_input_catches_an_auth_check_that_accepts_anything(self, monkeypatch):
        module, check = load("ui_input")
        monkeypatch.setattr(module, "validate_auth_shape", lambda _auth: None)
        assert first_failure(check, corpus_and_random("ui_input")) is not None

    def test_ui_input_catches_a_fence_that_never_grows(self, monkeypatch):
        module, check = load("ui_input")
        monkeypatch.setattr(
            module,
            "safe_markdown_fence",
            lambda content, language="": f"```{language}\n{content}\n```",
        )
        assert first_failure(check, corpus_and_random("ui_input")) is not None

    def test_ui_input_catches_a_bound_check_that_returns_out_of_range(self, monkeypatch):
        module, check = load("ui_input")
        monkeypatch.setattr(module, "validate_numeric_input", lambda value, *_a, **_k: 10.0**9)
        assert first_failure(check, corpus_and_random("ui_input")) is not None

    def test_datasets_catches_a_converter_that_loses_roles(self, monkeypatch):
        module, check = load("datasets")
        monkeypatch.setattr(module.FormatConverter, "ROLE_MAP_SHAREGPT", {})
        assert first_failure(check, corpus_and_random("datasets")) is not None

    def test_dataset_files_catches_a_loader_that_drops_a_row(self, monkeypatch):
        module, check = load("dataset_files")
        real = module.DatasetLoader

        class Lossy(real):
            @property
            def samples(self):
                return list(self._samples[:-1])

        monkeypatch.setattr(module, "DatasetLoader", Lossy)
        assert first_failure(check, corpus_and_random("dataset_files", 200)) is not None


# Seeds that reproduce a known finding: (target, file, exception under strict=True)
STRICT_SEEDS = [
    ("datasets", "finding_f6_lone_surrogate.seed", UnicodeEncodeError),
    ("ui_input", "finding_f4_nan.seed", AssertionError),
    ("config", "finding_f4b_orpo_beta.seed", AssertionError),
    ("config", "finding_f4b_simpo_gamma.seed", AssertionError),
]


class TestKnownFindings:
    """Bugs the fuzzers found. NOT fixed here; see the PR description.

    When one is fixed, its xfail test XPASSes (strict) -- delete the marker, and
    delete the matching tolerance in the harness (``strict`` branch) too.
    """

    @pytest.mark.parametrize(("target", "name", "exc_type"), STRICT_SEEDS)
    def test_strict_mode_rejects_the_reproducer_seed(self, target, name, exc_type):
        _, check = load(target)
        data = (CORPUS / target / name).read_bytes()
        check(data, strict=False)  # tolerated by default, so fuzzing can go on
        with pytest.raises(exc_type):
            check(data, strict=True)

    @pytest.mark.xfail(strict=True, reason="F4: NaN slips through validate_numeric_input bounds")
    def test_f4_validate_numeric_input_rejects_nan_against_a_range(self):
        from backpropagate.exceptions import UserInputError
        from backpropagate.ui_security import validate_numeric_input

        with pytest.raises(UserInputError):
            validate_numeric_input("nan", "learning_rate", min_value=0.0, max_value=1.0)

    @pytest.mark.xfail(strict=True, reason="F4b: NaN passes the 'reject non-positive' validators")
    @pytest.mark.parametrize("name", ["orpo_beta", "simpo_gamma", "kto_desirable_weight"])
    def test_f4b_training_config_rejects_nan_in_positive_fields(self, name):
        pytest.importorskip("pydantic_settings")
        from backpropagate.config import TrainingConfig

        with pytest.raises(Exception, match=r"(?i)positive|nan|invalid|setting"):
            TrainingConfig(**{name: float("nan")})

    @pytest.mark.xfail(raises=OverflowError, strict=True, reason="F5: huge int escapes as OverflowError")
    def test_f5_validate_numeric_input_wraps_an_overflowing_int(self):
        from backpropagate.exceptions import UserInputError
        from backpropagate.ui_security import validate_numeric_input

        with pytest.raises(UserInputError):
            validate_numeric_input(10**400, "n")

    @pytest.mark.xfail(raises=UnicodeEncodeError, strict=True, reason="F6: lone surrogate breaks dedupe")
    def test_f6_deduplicate_exact_survives_a_lone_surrogate(self):
        from backpropagate.datasets import deduplicate_exact

        rows = [{"text": "\ud800 broken"}, {"text": "fine"}]
        unique, removed = deduplicate_exact(rows)
        assert len(unique) + removed == 2

    @pytest.mark.xfail(
        sys.version_info < (3, 13),
        raises=RuntimeError,
        strict=True,
        reason="F7: safe_path leaks RuntimeError on a symlink loop (pathlib raises before 3.13)",
    )
    def test_f7_safe_path_turns_a_symlink_loop_into_a_clean_error(self, tmp_path):
        from backpropagate.security import PathTraversalError, safe_path

        base = tmp_path / "base"
        base.mkdir()
        try:
            os.symlink("loop_b", base / "loop_a")
            os.symlink("loop_a", base / "loop_b")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        try:
            safe_path(base / "loop_a", allowed_base=base)
        except (PathTraversalError, FileNotFoundError):
            pass

    @pytest.mark.xfail(strict=True, reason="F8: a space in the user name leaves the surname in the text")
    def test_f8_redact_paths_hides_a_two_word_user_name(self):
        from backpropagate.ui_security import _redact_paths

        out = _redact_paths(r"cannot open C:\Users\John Qzx9\data\x.jsonl")
        assert "Qzx9" not in out
        out = _redact_paths("cannot open /home/john qzx9/data/x.jsonl")
        assert "qzx9" not in out

    @pytest.mark.xfail(raises=RecursionError, strict=True, reason="F9: streaming loader leaks RecursionError")
    def test_f9_streaming_loader_wraps_deeply_nested_json(self, tmp_path):
        from backpropagate.datasets import StreamingDatasetLoader

        path = tmp_path / "deep.jsonl"
        path.write_text("[" * 20000 + "]" * 20000 + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            list(StreamingDatasetLoader(str(path)))


class TestFixedFindings:
    """Regression tests for bugs the fuzzers found and that are now fixed."""

    @pytest.mark.parametrize("role", [["user"], {"a": 1}, 7, None, "robot"])
    def test_f1_openai_role_that_is_not_a_known_string_is_a_validation_error(self, role):
        from backpropagate.datasets import validate_dataset

        result = validate_dataset([{"messages": [{"role": role, "content": "x"}]}])
        # An unknown role is reported as a warning (same as a string like "robot").
        assert [w.error_type for w in result.warnings] == ["invalid_role"]

    def test_f1_dataset_loader_reports_an_unhashable_role_instead_of_crashing(self, tmp_path):
        from backpropagate.datasets import DatasetLoader

        path = tmp_path / "bad_role.jsonl"
        path.write_text('{"messages": [{"role": ["user"], "content": "x"}]}\n', encoding="utf-8")
        loader = DatasetLoader(path)
        assert loader.validation_result.warnings[0].error_type == "invalid_role"


    @pytest.mark.parametrize(
        "name",
        ["\x01..\x01", "\x01.\x01", "..\x00..", " \x7f. ", "\x85.\x85.", "..", ".", "\x1f \x1f"],
    )
    def test_f2_sanitize_filename_never_returns_a_dot_component(self, name):
        from backpropagate.ui_security import sanitize_filename

        assert sanitize_filename(name) not in {".", ".."}
        assert sanitize_filename("\x01..\x01") == "unnamed_file"


    def test_f3_sanitize_filename_honours_its_length_limit(self):
        from backpropagate.ui_security import sanitize_filename

        out = sanitize_filename("x." + "a" * 300)  # the reproducer: a 301-char "extension"
        assert 0 < len(out) <= 255
        assert len(sanitize_filename("." + "b" * 400 + "." + "c" * 400)) <= 255

    @pytest.mark.parametrize("ext", [".jsonl", ".safetensors", ".gz"])
    def test_f3_a_long_name_keeps_its_extension(self, ext):
        from backpropagate.ui_security import sanitize_filename

        out = sanitize_filename("n" * 400 + ext)
        assert len(out) == 255
        assert out.endswith(ext)
        assert out.startswith("nnnn")

    def test_f3_truncation_does_not_leave_a_trailing_dot_or_space(self):
        from backpropagate.ui_security import sanitize_filename

        out = sanitize_filename("a" + "." * 400 + "b")
        assert len(out) <= 255
        assert not out.endswith((".", " "))


class TestPlumbing:
    def test_harnesses_import_without_atheris(self):
        # Importing a harness must not need Atheris (it has no Windows wheels).
        for target in TARGETS:
            load(target)

    def test_main_explains_itself_when_atheris_is_missing(self, monkeypatch, capsys):
        common = importlib.import_module("fuzz_common")
        monkeypatch.setattr(common, "atheris", None)
        assert common.main(lambda _data: None) == 2
        assert "requirements/fuzz.txt" in capsys.readouterr().err

    def test_provider_is_deterministic_and_total(self):
        common = importlib.import_module("fuzz_common")
        a, b = common.Provider(b"\x05abc\xff"), common.Provider(b"\x05abc\xff")
        assert [a.int_in_range(0, 9), a.text(4), a.bool()] == [
            b.int_in_range(0, 9),
            b.text(4),
            b.bool(),
        ]
        empty = common.Provider(b"")
        assert empty.int_in_range(3, 9) == 3
        assert empty.text() == "" and empty.bytes(5) == b"" and empty.pick("xyz") == "x"
        assert common.Provider(b"\xff").int_in_range(0, 4) in range(5)

    def test_instrumentation_is_off_unless_fuzzing(self):
        common = importlib.import_module("fuzz_common")
        with common.instrument(False):
            pass

    def test_patched_environ_restores_the_environment(self):
        common = importlib.import_module("fuzz_common")
        before = dict(os.environ)
        with pytest.raises(RuntimeError), common.patched_environ({"BP_FUZZ_TEST": "1", "PATH": "x"}):
            assert os.environ["BP_FUZZ_TEST"] == "1"
            raise RuntimeError("boom")
        assert dict(os.environ) == before

    # -- the workflow and the requirements file ------------------------------

    @property
    def workflow(self) -> str:
        return (ROOT / ".github" / "workflows" / "fuzz.yml").read_text(encoding="utf-8")

    def test_workflow_is_manual_dispatch_only(self):
        text = self.workflow
        on_block = re.search(r"^on:\n((?:[ \t]+.*\n|\n)+)", text, re.MULTILINE)
        assert on_block, "no top-level 'on:' block"
        triggers = set(re.findall(r"^  ([a-z_]+):", on_block.group(1), re.MULTILINE))
        assert triggers == {"workflow_dispatch"}, triggers

    def test_workflow_matches_the_repo_rules(self):
        text = self.workflow
        assert "runs-on: ubuntu-latest" in text
        assert re.search(r"timeout-minutes:\s*\d+", text)
        assert re.search(r"^concurrency:", text, re.MULTILINE)
        assert re.search(r"^permissions:\n  contents: read\n", text, re.MULTILINE)
        assert "pip install --require-hashes -r requirements/fuzz.txt" in text

    def test_every_action_is_pinned_to_a_commit_sha(self):
        uses = re.findall(r"^\s*(?:-\s*)?uses:\s*(\S+)", self.workflow, re.MULTILINE)
        assert uses, "the workflow uses no actions?"
        for ref in uses:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), f"not SHA-pinned: {ref}"

    def test_requirements_are_hash_pinned(self):
        text = (ROOT / "requirements" / "fuzz.txt").read_text(encoding="utf-8")
        entries = re.findall(r"^([A-Za-z0-9_.\-]+)==\S+ \\\n((?:\s+--hash=sha256:[0-9a-f]{64}.*\n)+)", text, re.MULTILINE)
        names = {name.lower() for name, _ in entries}
        assert {"atheris", "pydantic", "pydantic-settings", "tenacity", "filelock", "packaging"} <= names
        pinned_lines = [ln for ln in text.splitlines() if re.match(r"^[A-Za-z0-9_.\-]+==", ln)]
        assert len(pinned_lines) == len(entries), "a requirement has no --hash"
