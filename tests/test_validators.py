"""Regression tests for input validators in :mod:`backpropagate.export`.

These pin BRIDGE-A-001 (``_validate_model_name``) and BRIDGE-A-002
(``_validate_repo_id``) — both Wave 1 hardening fixes that previously
forwarded operator-supplied strings into subprocess argv / HTTP calls
without validation.

TESTS-B-002 escalation: before this file the validators had zero
regression tests. A future refactor (e.g. relaxing the regex, dropping
the explicit '..' check, or removing the leading-dash guard) would have
re-opened the option-injection / path-traversal surface without any CI
signal. Each test below names the adversarial category it pins.

Each test runs the validator directly (no subprocess / no network) so it
is fast, deterministic, and OS-independent. Both validators raise
``ExportError`` (a subclass of ``BackpropagateError``) with a stable
``code`` attribute the CLI scrapes for error formatting.
"""

import pytest

from backpropagate.exceptions import BackpropagateError, ExportError
from backpropagate.export import _validate_model_name, _validate_repo_id

# =============================================================================
# _validate_model_name — BRIDGE-A-001
# =============================================================================


class TestValidateModelName:
    """Pin the Ollama model-name allowlist contract."""

    # -- happy path -----------------------------------------------------------

    @pytest.mark.parametrize(
        "name",
        [
            "my-model",
            "qwen2.5-7b",
            "alice_finetune",
            "llama3.2:1b",
            "model-with.dots_and-dashes:v1",
        ],
    )
    def test_accepts_well_formed_name(self, name):
        """Well-formed model names should validate silently (return None).

        Note: ``org/model:tag`` (slash form) is REJECTED by the current
        ``_validate_model_name`` regex + Path-separator defense — Ollama
        names with ``/`` would hit the ``Path(model_name).name !=
        model_name`` check. The validator's contract is single-segment
        names with optional ``:tag``; ``/``-shaped names belong to the
        repo_id validator.
        """
        # No exception raised → contract satisfied.
        assert _validate_model_name(name) is None

    # -- empty / wrong type ---------------------------------------------------

    def test_rejects_empty_string(self):
        with pytest.raises(ExportError, match="non-empty string") as exc_info:
            _validate_model_name("")
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED"

    def test_rejects_non_string(self):
        with pytest.raises(ExportError, match="non-empty string") as exc_info:
            _validate_model_name(None)  # type: ignore[arg-type]
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED"

    def test_rejects_integer(self):
        with pytest.raises(ExportError, match="non-empty string") as exc_info:
            _validate_model_name(12345)  # type: ignore[arg-type]
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED"

    # -- leading dash (option injection) --------------------------------------

    def test_rejects_leading_dash(self):
        """``-h`` would otherwise be parsed as a flag by the ollama CLI."""
        with pytest.raises(ExportError, match="starts with '-'") as exc_info:
            _validate_model_name("-h")
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED"

    def test_rejects_leading_dash_longer_name(self):
        with pytest.raises(ExportError, match="starts with '-'"):
            _validate_model_name("--rm")

    # -- control characters ---------------------------------------------------

    def test_rejects_embedded_nul(self):
        with pytest.raises(ExportError, match="control character") as exc_info:
            _validate_model_name("foo\x00bar")
        assert getattr(exc_info.value, "code", None) == "INPUT_VALIDATION_FAILED"

    def test_rejects_embedded_newline(self):
        with pytest.raises(ExportError, match="control character"):
            _validate_model_name("foo\nbar")

    def test_rejects_embedded_carriage_return(self):
        with pytest.raises(ExportError, match="control character"):
            _validate_model_name("foo\rbar")

    # -- whitespace -----------------------------------------------------------

    def test_rejects_internal_space(self):
        """Whitespace inside the name would split the argv on the CLI side."""
        with pytest.raises(ExportError, match="invalid characters"):
            _validate_model_name("my model")

    def test_rejects_internal_tab(self):
        with pytest.raises(ExportError, match="invalid characters"):
            _validate_model_name("my\tmodel")

    # -- path separators (defense in depth) -----------------------------------

    def test_rejects_backslash(self):
        """Backslash is a Windows path separator; blocked by regex."""
        with pytest.raises(ExportError, match="invalid characters"):
            _validate_model_name("alice\\model")

    # -- oversize -------------------------------------------------------------

    def test_rejects_oversize_name(self):
        """Length cap is 128 chars (regex allows 0..127 after the first)."""
        oversize = "a" + ("b" * 200)  # 201 chars total
        with pytest.raises(ExportError, match="invalid characters"):
            _validate_model_name(oversize)

    # -- unicode --------------------------------------------------------------

    def test_rejects_unicode_rtl_override(self):
        """U+202E (right-to-left override) is a known display-spoofing char."""
        # The leading 'a' makes the first-char-alnum check pass; the U+202E
        # is what we want to reject via the allowlist.
        with pytest.raises(ExportError, match="invalid characters"):
            _validate_model_name("a‮model")

    def test_rejects_unicode_letter(self):
        """Allowlist is ASCII-only; non-ASCII letters are rejected."""
        with pytest.raises(ExportError, match="invalid characters"):
            _validate_model_name("café-model")


# =============================================================================
# _validate_repo_id — BRIDGE-A-002
# =============================================================================


class TestValidateRepoId:
    """Pin the Hugging Face repo_id 'owner/name' shape contract."""

    # -- happy path -----------------------------------------------------------

    @pytest.mark.parametrize(
        "repo_id",
        [
            "alice/qwen-finetune",
            "bob/model.v1",
            "org_name/some-model",
            "user123/model-2025",
            "AnthropicLabs/claude-adapter",
            "a/b",  # minimum well-formed shape
        ],
    )
    def test_accepts_well_formed_repo_id(self, repo_id):
        """Well-formed repo_ids should validate silently (return None)."""
        assert _validate_repo_id(repo_id) is None

    # -- empty / wrong type ---------------------------------------------------

    def test_rejects_empty_string(self):
        with pytest.raises(ExportError, match="non-empty string") as exc_info:
            _validate_repo_id("")
        assert getattr(exc_info.value, "code", None) == "HUB_PUSH_INVALID_REPO"

    def test_rejects_non_string(self):
        with pytest.raises(ExportError, match="non-empty string") as exc_info:
            _validate_repo_id(None)  # type: ignore[arg-type]
        assert getattr(exc_info.value, "code", None) == "HUB_PUSH_INVALID_REPO"

    def test_rejects_integer(self):
        with pytest.raises(ExportError, match="non-empty string"):
            _validate_repo_id(42)  # type: ignore[arg-type]

    # -- shape violations -----------------------------------------------------

    def test_rejects_missing_slash(self):
        with pytest.raises(ExportError, match="owner/name") as exc_info:
            _validate_repo_id("just-a-name")
        assert getattr(exc_info.value, "code", None) == "HUB_PUSH_INVALID_REPO"

    def test_rejects_double_separator(self):
        """``alice//foo`` produces an empty middle segment."""
        with pytest.raises(ExportError, match="owner/name") as exc_info:
            _validate_repo_id("alice//foo")
        assert getattr(exc_info.value, "code", None) == "HUB_PUSH_INVALID_REPO"

    def test_rejects_three_segments(self):
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id("alice/foo/bar")

    def test_rejects_leading_dash_owner(self):
        """First-char-alnum rule on each segment blocks leading dash."""
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id("-alice/model")

    def test_rejects_leading_dash_name(self):
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id("alice/-model")

    # -- path traversal -------------------------------------------------------

    def test_rejects_dotdot_owner(self):
        """``..`` would traverse out of the cached-readme mirror dir."""
        with pytest.raises(ExportError, match=r"'\.\.'") as exc_info:
            _validate_repo_id("../etc/passwd")
        assert getattr(exc_info.value, "code", None) == "HUB_PUSH_INVALID_REPO"

    def test_rejects_dotdot_segment(self):
        with pytest.raises(ExportError, match=r"'\.\.'"):
            _validate_repo_id("alice/../bob")

    def test_rejects_single_dot_segment(self):
        with pytest.raises(ExportError, match=r"'\.'"):
            _validate_repo_id("./model")

    # -- control characters + backslash ---------------------------------------

    def test_rejects_embedded_nul(self):
        with pytest.raises(ExportError, match="control character or backslash"):
            _validate_repo_id("alice/foo\x00bar")

    def test_rejects_embedded_newline(self):
        with pytest.raises(ExportError, match="control character or backslash"):
            _validate_repo_id("alice/foo\nbar")

    def test_rejects_embedded_carriage_return(self):
        with pytest.raises(ExportError, match="control character or backslash"):
            _validate_repo_id("alice/foo\rbar")

    def test_rejects_backslash(self):
        """Backslash inside repo_id is a Windows-path-traversal smell."""
        with pytest.raises(ExportError, match="control character or backslash"):
            _validate_repo_id("alice\\evil/bar")

    # -- whitespace -----------------------------------------------------------

    def test_rejects_internal_space(self):
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id("alice/my model")

    def test_rejects_leading_space(self):
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id(" alice/model")

    # -- oversize -------------------------------------------------------------

    def test_rejects_oversize_segment(self):
        """Each segment is capped at 96 chars."""
        oversize = "a" + ("b" * 200)  # 201 chars
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id(f"{oversize}/model")

    def test_rejects_oversize_name_segment(self):
        oversize = "a" + ("b" * 200)
        with pytest.raises(ExportError, match="owner/name"):
            _validate_repo_id(f"alice/{oversize}")

    # -- exception-hierarchy invariant ---------------------------------------

    def test_raised_error_is_backpropagate_error(self):
        """ExportError must be a BackpropagateError so the CLI catches it."""
        with pytest.raises(BackpropagateError):
            _validate_repo_id("")
