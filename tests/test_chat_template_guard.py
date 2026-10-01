"""Path-like chat template names are refused at load (PYSEC-2026-3929).

transformers before 5.10 saves each named chat template to
``additional_chat_templates/<name>.jinja`` without checking ``name``, so a
crafted tokenizer_config.json on the Hub can write outside the save directory.
unsloth's caps hold transformers at 5.5.0, so backpropagate checks the names
itself, right after every tokenizer load whose tokenizer can later be saved.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backpropagate.exceptions import ERROR_CODES, ModelLoadError, UnsafeChatTemplateError
from backpropagate.security import check_chat_template_names
from tests.helpers.tiny_models import tiny_tokenizer


def _tok(chat_template):
    return SimpleNamespace(chat_template=chat_template)


class TestCheck:
    @pytest.mark.parametrize(
        "template",
        [None, "{{ messages }}", {"default": "a", "tool_use": "b", "rag": "c", "v1.2-x": "d"}],
    )
    def test_plain_names_pass(self, template):
        check_chat_template_names(_tok(template), "org/model")

    @pytest.mark.parametrize(
        "name",
        ["../escape", "../../etc/x", "a/b", "a\\b", "/abs", "C:\\x", "C:x", "..", ".hidden", "", "x" * 200],
    )
    def test_path_like_names_refused(self, name):
        with pytest.raises(UnsafeChatTemplateError) as exc:
            check_chat_template_names(_tok({"default": "a", name: "b"}), "org/model")
        assert exc.value.names == [name]

    def test_non_string_name_refused(self):
        with pytest.raises(UnsafeChatTemplateError):
            check_chat_template_names(_tok({"default": "a", 3: "b"}), "org/model")

    def test_error_contract(self):
        err = UnsafeChatTemplateError("org/model", ["../x"])
        assert isinstance(err, ModelLoadError)
        assert err.code == "INPUT_UNSAFE_CHAT_TEMPLATE"
        assert err.retryable is False
        assert "INPUT_UNSAFE_CHAT_TEMPLATE" in ERROR_CODES
        assert "org/model" in str(err) and "../x" in str(err)


class TestRealTokenizer:
    def test_guard_refuses_what_transformers_would_write_outside(self, tmp_path):
        """A real tokenizer with a '../../../' template name. The guard refuses it;
        on an affected transformers (< 5.10) saving it writes a file outside the
        save directory, which is the write the guard exists to prevent."""
        import transformers
        from packaging.version import Version

        tok = tiny_tokenizer()
        tok.chat_template = {"default": "{{ messages }}", "../../../escaped": "pwned"}
        with pytest.raises(UnsafeChatTemplateError):
            check_chat_template_names(tok, "org/model")

        if Version(transformers.__version__) >= Version("5.10"):
            pytest.skip("transformers >= 5.10 validates template names itself")
        out = tmp_path / "a" / "b" / "save"
        out.mkdir(parents=True)
        tok.save_pretrained(str(out))
        escaped = list(tmp_path.rglob("escaped.jinja"))
        assert escaped and all(out not in p.parents for p in escaped)
        assert escaped[0].read_text(encoding="utf-8") == "pwned"


class TestLoadPaths:
    def test_export_loader_refuses(self, tmp_path, monkeypatch):
        """load_model_for_export checks the tokenizer it is about to hand to a save."""
        import transformers

        import backpropagate.export as ex

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                            classmethod(lambda cls, *a, **k: object()))
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            classmethod(lambda cls, *a, **k: _tok({"../x": "y"})))
        with pytest.raises(UnsafeChatTemplateError):
            ex.load_model_for_export(tmp_path)

    def test_trainer_load_paths_call_the_check(self):
        """Both Trainer load paths (Unsloth and transformers) call the check
        right after the tokenizer is loaded."""
        import inspect

        import backpropagate.trainer as t

        src = inspect.getsource(t)
        assert src.count("check_chat_template_names(self._tokenizer, self.model_name)") == 2
