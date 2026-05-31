"""Tests for export functions."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExportEnums:
    """Tests for export enums."""

    def test_gguf_quantization_values(self):
        """Test GGUFQuantization enum has expected values."""
        from backpropagate.export import GGUFQuantization

        assert GGUFQuantization.F16.value == "f16"
        assert GGUFQuantization.Q8_0.value == "q8_0"
        assert GGUFQuantization.Q5_K_M.value == "q5_k_m"
        assert GGUFQuantization.Q4_K_M.value == "q4_k_m"
        assert GGUFQuantization.Q4_0.value == "q4_0"
        assert GGUFQuantization.Q2_K.value == "q2_k"

    def test_export_format_values(self):
        """Test ExportFormat enum has expected values."""
        from backpropagate.export import ExportFormat

        assert ExportFormat.LORA.value == "lora"
        assert ExportFormat.MERGED.value == "merged"
        assert ExportFormat.GGUF.value == "gguf"


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_creation(self, temp_dir):
        """Test ExportResult can be created."""
        from backpropagate.export import ExportFormat, ExportResult

        result = ExportResult(
            format=ExportFormat.LORA,
            path=temp_dir / "model",
            size_mb=100.5,
            export_time_seconds=5.2,
        )

        assert result.format == ExportFormat.LORA
        assert result.size_mb == 100.5
        assert result.export_time_seconds == 5.2

    def test_export_result_summary_lora(self, temp_dir):
        """Test ExportResult summary for LoRA format."""
        from backpropagate.export import ExportFormat, ExportResult

        result = ExportResult(
            format=ExportFormat.LORA,
            path=temp_dir / "model",
            size_mb=100.5,
            export_time_seconds=5.2,
        )

        summary = result.summary()
        assert "lora" in summary.lower()
        assert "100.5" in summary
        assert "5.2" in summary

    def test_export_result_summary_gguf(self, temp_dir):
        """Test ExportResult summary for GGUF format."""
        from backpropagate.export import ExportFormat, ExportResult

        result = ExportResult(
            format=ExportFormat.GGUF,
            path=temp_dir / "model.gguf",
            size_mb=2048.0,
            quantization="q4_k_m",
            export_time_seconds=120.5,
        )

        summary = result.summary()
        assert "gguf" in summary.lower()
        assert "q4_k_m" in summary
        assert "2048.0" in summary


class TestExportLora:
    """Tests for export_lora function."""

    def test_export_lora_from_path(self, temp_dir):
        """Test exporting LoRA from a path."""
        from backpropagate.export import ExportFormat, export_lora

        # Create source adapter files
        src_dir = temp_dir / "source"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{"test": true}')
        (src_dir / "adapter_model.safetensors").write_bytes(b"mock safetensors")

        output_dir = temp_dir / "output"

        result = export_lora(model=src_dir, output_dir=output_dir)

        assert result.format == ExportFormat.LORA
        assert result.path == output_dir
        assert (output_dir / "adapter_config.json").exists()

    def test_export_lora_from_peft_model(self, temp_dir, mock_peft_model):
        """Test exporting LoRA from a PeftModel."""
        from backpropagate.export import ExportFormat, export_lora

        # Patch the peft check to recognize our mock
        with patch("backpropagate.export._is_peft_model", return_value=True):
            output_dir = temp_dir / "output"

            result = export_lora(model=mock_peft_model, output_dir=output_dir)

            assert result.format == ExportFormat.LORA
            mock_peft_model.save_pretrained.assert_called_once()

    def test_export_lora_invalid_type(self, temp_dir):
        """Test export_lora raises error for invalid model type."""
        # String paths that don't exist should raise ExportError
        from backpropagate.exceptions import ExportError
        from backpropagate.export import export_lora
        with patch("backpropagate.export._is_peft_model", return_value=False):
            with pytest.raises(ExportError, match="Cannot export LoRA"):
                export_lora(model=12345, output_dir=temp_dir)  # Non-path, non-model type


class TestExportMerged:
    """Tests for export_merged function."""

    def test_export_merged_basic(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test basic merged export."""
        from backpropagate.export import ExportFormat, export_merged

        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_merged(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "merged",
            )

        assert result.format == ExportFormat.MERGED
        mock_peft_model.merge_and_unload.assert_called_once()
        merged_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_export_merged_push_to_hub(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test merged export with push_to_hub."""
        from backpropagate.export import export_merged

        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        merged_model.push_to_hub = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_merged(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "merged",
                push_to_hub=True,
                repo_id="test/repo",
            )

        merged_model.push_to_hub.assert_called_once_with("test/repo")
        mock_tokenizer.push_to_hub.assert_called_once_with("test/repo")

    def test_export_merged_requires_repo_id(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test export_merged raises error when push_to_hub=True but no repo_id."""
        from backpropagate.exceptions import MergeExportError
        from backpropagate.export import export_merged

        with patch("backpropagate.export._is_peft_model", return_value=True):
            with pytest.raises(MergeExportError, match="repo_id required"):
                export_merged(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir,
                    push_to_hub=True,
                )

    def test_export_merged_invalid_model(self, temp_dir, mock_tokenizer):
        """Test export_merged raises error for non-PeftModel."""
        from backpropagate.exceptions import MergeExportError
        from backpropagate.export import export_merged

        with pytest.raises(MergeExportError, match="Cannot merge"):
            export_merged(
                model=MagicMock(),
                tokenizer=mock_tokenizer,
                output_dir=temp_dir,
            )


class TestExportGguf:
    """Tests for export_gguf function."""

    def test_export_gguf_with_unsloth(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test GGUF export using Unsloth."""
        from backpropagate.export import ExportFormat, export_gguf

        # Create a mock GGUF file that Unsloth would create
        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model-q4_k_m.gguf").write_bytes(b"mock gguf")

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "gguf",
                quantization="q4_k_m",
            )

        assert result.format == ExportFormat.GGUF
        assert result.quantization == "q4_k_m"

    def test_export_gguf_invalid_quantization(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test export_gguf raises error for invalid quantization."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.export import export_gguf

        with pytest.raises(InvalidSettingError, match="quantization"):
            export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir,
                quantization="invalid_quant",
            )

    def test_export_gguf_quantization_enum(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test export_gguf accepts GGUFQuantization enum."""
        from backpropagate.export import GGUFQuantization, export_gguf

        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model-q8_0.gguf").write_bytes(b"mock gguf")

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "gguf",
                quantization=GGUFQuantization.Q8_0,
            )

        assert result.quantization == "q8_0"


class TestModelfile:
    """Tests for Modelfile creation."""

    def test_create_modelfile_basic(self, sample_gguf_path):
        """Test basic Modelfile creation.

        BRIDGE-A-001 (v1.1.2 amend wave): create_modelfile now escapes
        backslash and double-quote in the FROM path so a gguf_path with
        either character (UNC / Windows paths) produces a well-formed
        Modelfile. We assert against the post-escape form to pin the new
        contract — a raw ``str(path)`` substring match would fail on
        Windows where ``C:\\Users\\...`` becomes ``C:\\\\Users\\\\...``
        inside the Modelfile.
        """
        from backpropagate.export import create_modelfile

        modelfile = create_modelfile(sample_gguf_path)

        assert modelfile.exists()
        content = modelfile.read_text()
        assert "FROM" in content
        # Build the expected escaped form the same way create_modelfile does:
        # escape backslashes first, then quotes (order matters — otherwise the
        # inserted escape-backslashes are themselves doubled).
        expected_escaped_path = (
            str(sample_gguf_path.resolve())
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )
        assert expected_escaped_path in content, (
            f"Escaped FROM path {expected_escaped_path!r} not found in Modelfile "
            f"content:\n{content!r}"
        )

    def test_create_modelfile_with_options(self, sample_gguf_path):
        """Test Modelfile creation with custom options."""
        from backpropagate.export import create_modelfile

        modelfile = create_modelfile(
            sample_gguf_path,
            system_prompt="You are a helpful assistant.",
            temperature=0.8,
            context_length=8192,
        )

        content = modelfile.read_text()
        assert "0.8" in content
        assert "8192" in content
        assert "helpful assistant" in content

    def test_create_modelfile_custom_output_path(self, temp_dir, sample_gguf_path):
        """Test Modelfile creation with custom output path."""
        from backpropagate.export import create_modelfile

        custom_path = temp_dir / "custom" / "Modelfile"
        custom_path.parent.mkdir(parents=True, exist_ok=True)

        modelfile = create_modelfile(sample_gguf_path, output_path=custom_path)

        assert modelfile == custom_path
        assert modelfile.exists()

    def test_create_modelfile_escapes_quotes(self, sample_gguf_path):
        """Test Modelfile properly escapes quotes in system prompt."""
        from backpropagate.export import create_modelfile

        modelfile = create_modelfile(
            sample_gguf_path,
            system_prompt='Say "hello" to the user.',
        )

        content = modelfile.read_text()
        assert '\\"hello\\"' in content

    @pytest.mark.parametrize(
        "ctrl",
        ["\x0c", "\x0b", "\x07", "\x01", "\x1f", "\x08", "\x1b"],
        ids=["form-feed", "vtab", "bell", "soh", "unit-sep", "backspace", "esc"],
    )
    def test_create_modelfile_rejects_c0_controls_in_system_prompt(self, sample_gguf_path, ctrl):
        """DATA-A-009: every C0 control (except TAB) in system_prompt is rejected.

        Pre-fix only NUL / newline / CR were caught; form-feed, vertical-tab
        and the other C0 controls slipped into the SYSTEM directive where some
        parsers treat them as line breaks / separators.
        """
        from backpropagate.exceptions import ExportError
        from backpropagate.export import create_modelfile

        with pytest.raises(ExportError) as exc_info:
            create_modelfile(sample_gguf_path, system_prompt=f"hello{ctrl}world")
        assert exc_info.value.code == "INPUT_VALIDATION_FAILED"

    def test_create_modelfile_still_rejects_nul_newline_cr(self, sample_gguf_path):
        """DATA-A-009 regression guard: the original NUL/LF/CR checks survive."""
        from backpropagate.exceptions import ExportError
        from backpropagate.export import create_modelfile

        for ctrl in ("\x00", "\n", "\r"):
            with pytest.raises(ExportError):
                create_modelfile(sample_gguf_path, system_prompt=f"a{ctrl}b")

    def test_create_modelfile_allows_tab_in_system_prompt(self, sample_gguf_path):
        """DATA-A-009: TAB is legitimate whitespace and must NOT be rejected."""
        from backpropagate.export import create_modelfile

        modelfile = create_modelfile(
            sample_gguf_path,
            system_prompt="role:\tassistant",
        )
        assert modelfile.exists()
        assert "\t" in modelfile.read_text()


class TestOllamaIntegration:
    """Tests for Ollama integration functions."""

    def test_register_with_ollama_file_not_found(self, temp_dir):
        """Test register_with_ollama raises error for missing file."""
        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import register_with_ollama

        with pytest.raises(OllamaRegistrationError, match="GGUF file not found"):
            register_with_ollama(temp_dir / "nonexistent.gguf", "test-model")

    def test_register_with_ollama_no_ollama(self, sample_gguf_path):
        """Test register_with_ollama raises error when Ollama not found."""
        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import register_with_ollama

        with patch("shutil.which", return_value=None):
            with pytest.raises(OllamaRegistrationError, match="Ollama CLI not found"):
                register_with_ollama(sample_gguf_path, "test-model")

    def test_register_with_ollama_success(self, sample_gguf_path):
        """Test successful Ollama registration.

        BRIDGE-A-004 (v1.1.2 amend wave): register_with_ollama now calls
        ``_run_subprocess_interruptible`` (Popen-based) instead of
        ``subprocess.run``, so Ctrl+C reliably propagates to the child
        ollama process instead of leaving a zombie. The mock target moved
        from ``subprocess.run`` to ``backpropagate.export._run_subprocess_interruptible``.
        """
        import subprocess as _sp

        from backpropagate.export import register_with_ollama

        mock_result = _sp.CompletedProcess(
            args=["ollama", "create", "test-model"],
            returncode=0,
            stdout="",
            stderr="",
        )

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch(
                 "backpropagate.export._run_subprocess_interruptible",
                 return_value=mock_result,
             ) as mock_run:
            result = register_with_ollama(sample_gguf_path, "test-model")

        assert result is True
        mock_run.assert_called_once()

    def test_register_with_ollama_failure(self, sample_gguf_path):
        """Test Ollama registration failure raises OllamaRegistrationError.

        BRIDGE-A-004 (v1.1.2 amend wave): register_with_ollama now invokes
        ``_run_subprocess_interruptible`` (Popen-based) instead of
        ``subprocess.run``. Patching the legacy ``subprocess.run`` target
        was a no-op — the test only passed on machines where ``ollama``
        happened to be installed AND happened to fail with the expected
        signature. On a clean CI runner without ollama, Popen raised
        ``FileNotFoundError`` and the assertion would never match.

        TESTS-B-001 escalation fix: move the patch target to
        ``backpropagate.export._run_subprocess_interruptible`` so the
        synthesized ``CalledProcessError`` actually reaches the
        ``OllamaRegistrationError`` translation branch — pinning the
        failure-message contract independent of host ollama state.
        """
        import subprocess

        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import register_with_ollama

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch(
                 "backpropagate.export._run_subprocess_interruptible",
                 side_effect=subprocess.CalledProcessError(1, "ollama"),
             ):
            with pytest.raises(OllamaRegistrationError, match="ollama create failed"):
                register_with_ollama(sample_gguf_path, "test-model")

    def test_list_ollama_models_no_ollama(self):
        """Test list_ollama_models returns empty when Ollama not found."""
        from backpropagate.export import list_ollama_models

        with patch("shutil.which", return_value=None):
            models = list_ollama_models()

        assert models == []

    def test_list_ollama_models_success(self):
        """Test list_ollama_models returns model names."""
        from backpropagate.export import list_ollama_models

        mock_result = MagicMock()
        mock_result.stdout = "NAME                 ID              SIZE     MODIFIED\nllama2:latest        abc123          3.8GB    1 day ago\nmistral:latest       def456          4.1GB    2 days ago\n"
        mock_result.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", return_value=mock_result):
            models = list_ollama_models()

        assert "llama2:latest" in models
        assert "mistral:latest" in models

    def test_list_ollama_models_error(self):
        """Test list_ollama_models returns empty on error."""
        import subprocess

        from backpropagate.export import list_ollama_models

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ollama")):
            models = list_ollama_models()

        assert models == []


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_dir_size_file(self, temp_dir):
        """Test _get_dir_size_mb for a single file."""
        from backpropagate.export import _get_dir_size_mb

        # Create a 1MB file
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"x" * (1024 * 1024))

        size = _get_dir_size_mb(test_file)
        assert 0.9 < size < 1.1  # Approximately 1 MB

    def test_get_dir_size_directory(self, temp_dir):
        """Test _get_dir_size_mb for a directory."""
        from backpropagate.export import _get_dir_size_mb

        # Create multiple files
        (temp_dir / "file1.bin").write_bytes(b"x" * (512 * 1024))
        (temp_dir / "file2.bin").write_bytes(b"x" * (512 * 1024))

        size = _get_dir_size_mb(temp_dir)
        assert 0.9 < size < 1.1  # Approximately 1 MB total

    def test_is_peft_model_true(self):
        """_is_peft_model returns True for an actual PeftModel instance.

        TESTS-A-010 (v1.3 Wave 1): the previous version of this test was a
        false-pass — it created an empty ``with`` block that did nothing,
        then asserted ``isinstance(result, bool)`` which is trivially true
        for both True and False return paths. The test passed regardless of
        whether ``_is_peft_model`` worked.

        New shape: patch the ``peft.PeftModel`` class the function imports
        and pass a real instance of the patched class so the inner
        ``isinstance(model, PeftModel)`` returns True. A regression that
        broke the import (e.g. moved the symbol) or the isinstance check
        (e.g. typo'd the type name) now fails this test.
        """
        from backpropagate.export import _is_peft_model

        # Use a real local class so isinstance() against it is meaningful.
        # This is intentionally NOT a MagicMock — MagicMock instances satisfy
        # ``isinstance(x, MagicMock)`` regardless of the class under test,
        # which would re-create the false-pass shape.
        class FakePeftModel:
            pass

        instance = FakePeftModel()

        # Patch the symbol the function imports inside its try block. The
        # function does ``from peft import PeftModel`` then
        # ``isinstance(model, PeftModel)`` — so we need to make the import
        # resolve to FakePeftModel.
        import sys
        import types

        fake_peft_module = types.ModuleType("peft")
        fake_peft_module.PeftModel = FakePeftModel  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"peft": fake_peft_module}):
            result = _is_peft_model(instance)
        # Active assertion — the FakePeftModel-typed instance MUST satisfy
        # the isinstance check inside _is_peft_model.
        assert result is True, (
            "_is_peft_model returned False for an instance of the patched "
            "PeftModel class — the isinstance check inside the function "
            "is broken or the import is resolving wrong."
        )

    def test_is_peft_model_false_for_unrelated_type(self):
        """_is_peft_model returns False for objects that are NOT PeftModel.

        Companion to test_is_peft_model_true above — pins the negative
        path so a regression that always returns True (e.g. ``return
        True`` slipped in for debugging) is caught.
        """
        from backpropagate.export import _is_peft_model

        class FakePeftModel:
            pass

        class UnrelatedModel:
            pass

        import sys
        import types

        fake_peft_module = types.ModuleType("peft")
        fake_peft_module.PeftModel = FakePeftModel  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"peft": fake_peft_module}):
            result = _is_peft_model(UnrelatedModel())
        assert result is False

    def test_is_peft_model_false_when_peft_missing(self):
        """_is_peft_model returns False when peft is not installed.

        Pins the ImportError-handling branch so a regression that lets
        ImportError propagate is caught immediately.
        """
        # Make the import inside the function fail by injecting a None
        # entry into sys.modules (sentinel for "import failed").
        import sys

        from backpropagate.export import _is_peft_model

        with patch.dict(sys.modules, {"peft": None}):
            result = _is_peft_model(object())
        assert result is False

    def test_has_unsloth(self):
        """Test _has_unsloth detection — exercises both branches.

        TESTS-B-009 (Stage C humanization): the prior implementation had a
        dead ``with patch(... ImportError ...): pass`` block that asserted
        nothing for the import-fail branch — the actual ``_has_unsloth()``
        call ran OUTSIDE both patch contexts, so the appearance-of-coverage
        for the ImportError path was theatrical. This rewrite runs the call
        INSIDE the patch so the False branch is actually exercised AND
        forces the unsloth module to be removed from sys.modules so the
        ``import unsloth`` statement inside ``_has_unsloth`` actually
        triggers the import machinery (rather than hitting the module cache).
        """
        import sys

        from backpropagate.export import _has_unsloth

        # Import-fail branch: force the unsloth import to raise ImportError.
        # Must (a) clear the module cache so the import statement re-runs
        # the import machinery (otherwise ``import unsloth`` is a fast
        # ``sys.modules['unsloth']`` lookup and the patched __import__ is
        # never consulted), and (b) raise ImportError from __import__ for
        # 'unsloth' and 'unsloth.*' subpackages.
        real_import = __import__
        cached_unsloth_keys = [k for k in list(sys.modules) if k == "unsloth" or k.startswith("unsloth.")]
        cached_unsloth = {k: sys.modules.pop(k) for k in cached_unsloth_keys}

        def _no_unsloth(name, *args, **kwargs):
            if name == "unsloth" or name.startswith("unsloth."):
                raise ImportError("Patched: no unsloth in this test")
            return real_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=_no_unsloth):
                assert _has_unsloth() is False, (
                    "_has_unsloth() must return False when the unsloth import "
                    "raises ImportError (the patch fired but the function "
                    "still claimed unsloth was available — coverage bug)"
                )
        finally:
            # Restore module cache so downstream tests in this session that
            # use unsloth aren't impacted.
            sys.modules.update(cached_unsloth)

        # Real-environment branch: now call without patches to verify the
        # function returns a bool either way (this is what the prior test
        # actually asserted; we keep it as a smoke check + the platform
        # skips for hosts where unsloth import raises non-ImportError).
        try:
            result = _has_unsloth()
            assert isinstance(result, bool)
        except (RuntimeError, Exception) as e:
            # May fail on Python 3.14 (torch.compile) or CI (no GPU)
            error_msg = str(e).lower()
            if "torch.compile" in error_msg or "3.14" in error_msg:
                pytest.skip("Unsloth incompatible with Python 3.14")
            elif "accelerator" in error_msg or "gpu" in error_msg:
                pytest.skip("Unsloth requires GPU")
            else:
                raise


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestExportGgufWithoutUnsloth:
    """Tests for export_gguf fallback path without Unsloth."""

    def test_export_gguf_no_unsloth_no_llama_cpp_raises(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should raise GGUFExportError when no Unsloth and no llama.cpp."""

        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        # Create merged model mock
        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        with patch("backpropagate.export._has_unsloth", return_value=False), \
             patch("backpropagate.export._is_peft_model", return_value=True):
            # No llama.cpp convert script exists
            with pytest.raises(GGUFExportError, match="GGUF export requires"):
                export_gguf(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "gguf",
                    quantization="q4_k_m",
                )

    def test_export_gguf_fallback_non_peft_model(self, temp_dir, mock_tokenizer):
        """export_gguf fallback should handle non-PEFT models."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        # Non-PEFT model (base model)
        base_model = MagicMock()
        base_model.save_pretrained = MagicMock()

        with patch("backpropagate.export._has_unsloth", return_value=False), \
             patch("backpropagate.export._is_peft_model", return_value=False):
            # Should attempt to use the model directly (without merging)
            with pytest.raises(GGUFExportError, match="GGUF export requires"):
                export_gguf(
                    model=base_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "gguf",
                    quantization="q4_k_m",
                )

    def test_export_gguf_unsloth_fails_falls_back(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should fall back when Unsloth export fails."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        # Unsloth save fails
        mock_peft_model.save_pretrained_gguf = MagicMock(
            side_effect=Exception("Unsloth export failed")
        )

        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        with patch("backpropagate.export._has_unsloth", return_value=True), \
             patch("backpropagate.export._is_peft_model", return_value=True):
            # Falls back but no llama.cpp available
            with pytest.raises(GGUFExportError, match="GGUF export requires"):
                export_gguf(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "gguf",
                    quantization="q4_k_m",
                )


class TestExportLoraFromPath:
    """Tests for export_lora from existing path."""

    def test_export_lora_from_string_path(self, temp_dir):
        """export_lora should accept string path to existing adapter."""
        from backpropagate.export import ExportFormat, export_lora

        # Create source adapter files
        src_dir = temp_dir / "source_adapter"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{"test": true}')
        (src_dir / "adapter_model.safetensors").write_bytes(b"mock safetensors data")

        output_dir = temp_dir / "output_adapter"

        # Pass as string
        result = export_lora(model=str(src_dir), output_dir=output_dir)

        assert result.format == ExportFormat.LORA
        assert (output_dir / "adapter_config.json").exists()
        assert (output_dir / "adapter_model.safetensors").exists()

    def test_export_lora_copies_bin_files(self, temp_dir):
        """export_lora should copy .bin adapter files."""
        from backpropagate.export import export_lora

        # Create source with .bin files (older format)
        src_dir = temp_dir / "source_bin"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{"format": "bin"}')
        (src_dir / "adapter_model.bin").write_bytes(b"mock bin data")

        output_dir = temp_dir / "output_bin"

        result = export_lora(model=src_dir, output_dir=output_dir)

        assert (output_dir / "adapter_config.json").exists()
        assert (output_dir / "adapter_model.bin").exists()

    def test_export_lora_preserves_file_contents(self, temp_dir):
        """export_lora should preserve exact file contents."""
        from backpropagate.export import export_lora

        src_dir = temp_dir / "source"
        src_dir.mkdir()
        config_content = '{"r": 16, "lora_alpha": 32}'
        (src_dir / "adapter_config.json").write_text(config_content)

        output_dir = temp_dir / "output"

        export_lora(model=src_dir, output_dir=output_dir)

        result_content = (output_dir / "adapter_config.json").read_text()
        assert result_content == config_content


class TestFindGgufFile:
    """Tests for GGUF file finding after export."""

    def test_export_gguf_finds_generated_file(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should find the generated GGUF file."""
        from backpropagate.export import ExportFormat, export_gguf

        # Create multiple GGUF files that might be generated
        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            # Unsloth might generate files with different naming patterns
            (Path(path) / "model-unsloth-q4_k_m.gguf").write_bytes(b"mock gguf 1")
            (Path(path) / "model-q4_k_m.gguf").write_bytes(b"mock gguf 2")

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "gguf",
                quantization="q4_k_m",
            )

        assert result.format == ExportFormat.GGUF
        # Should find one of the GGUF files
        assert result.path.suffix == ".gguf"

    def test_export_gguf_uses_model_name_when_no_gguf_found(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should raise GGUFExportError when no GGUF file is created."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        # Save doesn't create any GGUF file
        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            # No GGUF file created

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        with patch("backpropagate.export._has_unsloth", return_value=True):
            with pytest.raises(GGUFExportError, match="GGUF file was not created"):
                export_gguf(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "gguf",
                    quantization="q4_k_m",
                    model_name="my-custom-model",
                )

    def test_export_gguf_picks_first_gguf_from_multiple(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should pick first GGUF when multiple exist."""
        from backpropagate.export import export_gguf

        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            # Multiple GGUF files
            (Path(path) / "aaa-first.gguf").write_bytes(b"mock gguf 1")
            (Path(path) / "bbb-second.gguf").write_bytes(b"mock gguf 2")
            (Path(path) / "ccc-third.gguf").write_bytes(b"mock gguf 3")

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "gguf",
                quantization="q4_k_m",
            )

        # Should find and use one of the GGUF files
        assert result.path.exists()
        assert result.path.suffix == ".gguf"


class TestExportGgufModelNameSanitization:
    """DATA-A-004: export_gguf must not let model_name escape output_dir.

    The manual-conversion path interpolates model_name into the output GGUF
    filename. A model_name carrying path separators / ``..`` segments would
    otherwise write the GGUF outside output_dir (path traversal). The Ollama
    register/remove sinks already validate model names; this programmatic
    entry point did not.
    """

    def _run_manual_export(self, monkeypatch, temp_dir, model, tokenizer, model_name):
        """Drive the llama.cpp (manual) export path with a stub convert script.

        Returns the ExportResult. The stub ``_run_subprocess_interruptible``
        writes a real byte to whatever ``--outfile`` path export_gguf chose,
        so the produced file's location is observable.
        """
        from backpropagate import export as export_mod

        # Point the convert-script discovery at a real file via the env hatch.
        fake_convert = temp_dir / "convert_hf_to_gguf.py"
        fake_convert.write_text("# stub convert script")
        monkeypatch.setenv("BACKPROPAGATE_LLAMA_CPP_PATH", str(fake_convert))

        def fake_subprocess(cmd, **kwargs):
            # cmd == [python, script, merged, "--outfile", <gguf_path>, "--outtype", quant]
            outfile = Path(cmd[cmd.index("--outfile") + 1])
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(b"GGUF")
            import subprocess as _sp
            return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch.object(export_mod, "_has_unsloth", return_value=False), \
             patch.object(export_mod, "_is_peft_model", return_value=True), \
             patch.object(export_mod, "_run_subprocess_interruptible", side_effect=fake_subprocess):
            return export_mod.export_gguf(
                model=model,
                tokenizer=tokenizer,
                output_dir=temp_dir / "out",
                quantization="q4_k_m",
                model_name=model_name,
            )

    def test_traversal_model_name_stays_within_output_dir(
        self, monkeypatch, temp_dir, mock_peft_model, mock_tokenizer
    ):
        """A ``../`` model_name must produce a leaf file inside output_dir."""
        out_dir = (temp_dir / "out").resolve()
        result = self._run_manual_export(
            monkeypatch, temp_dir, mock_peft_model, mock_tokenizer,
            model_name="../../../evil",
        )

        # The produced GGUF is confined to output_dir.
        assert result.path.resolve().parent == out_dir
        assert result.path.exists()
        # The traversal target outside output_dir was NEVER created.
        assert not (temp_dir / "evil-q4_k_m.gguf").exists()
        assert not (temp_dir.parent / "evil-q4_k_m.gguf").exists()

    def test_separator_model_name_reduced_to_leaf(
        self, monkeypatch, temp_dir, mock_peft_model, mock_tokenizer
    ):
        """``sub/dir/x`` collapses to leaf ``x`` and stays in output_dir."""
        out_dir = (temp_dir / "out").resolve()
        result = self._run_manual_export(
            monkeypatch, temp_dir, mock_peft_model, mock_tokenizer,
            model_name="sub/dir/x",
        )
        assert result.path.resolve().parent == out_dir
        assert result.path.name == "x-q4_k_m.gguf"
        assert not (out_dir / "sub").exists()

    def test_clean_model_name_is_preserved(
        self, monkeypatch, temp_dir, mock_peft_model, mock_tokenizer
    ):
        """A normal model_name is used verbatim (no false-positive mangling)."""
        out_dir = (temp_dir / "out").resolve()
        result = self._run_manual_export(
            monkeypatch, temp_dir, mock_peft_model, mock_tokenizer,
            model_name="my-finetune",
        )
        assert result.path.name == "my-finetune-q4_k_m.gguf"
        assert result.path.resolve().parent == out_dir


class TestExportResultSummary:
    """Additional tests for ExportResult.summary() method."""

    def test_summary_without_time(self, temp_dir):
        """summary() should handle zero export time gracefully."""
        from backpropagate.export import ExportFormat, ExportResult

        result = ExportResult(
            format=ExportFormat.LORA,
            path=temp_dir / "model",
            size_mb=50.0,
            export_time_seconds=0.0,  # No time recorded
        )

        summary = result.summary()
        assert "50.0 MB" in summary
        assert "Time:" not in summary  # Should not show 0s time

    def test_summary_without_quantization(self, temp_dir):
        """summary() should handle missing quantization."""
        from backpropagate.export import ExportFormat, ExportResult

        result = ExportResult(
            format=ExportFormat.MERGED,
            path=temp_dir / "merged",
            size_mb=8000.0,
            quantization=None,
            export_time_seconds=60.0,
        )

        summary = result.summary()
        assert "Quantization" not in summary
        assert "merged" in summary.lower()


# =============================================================================
# TQ-001: PermissionError on output directory creation
# =============================================================================


class TestPermissionErrorOnOutputDir:
    """Tests for PermissionError when creating output directories (TQ-001)."""

    def test_export_lora_permission_error(self, temp_dir, mock_peft_model):
        """export_lora should raise ExportError when output dir can't be created."""
        from backpropagate.exceptions import ExportError
        from backpropagate.export import export_lora

        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(ExportError, match="Cannot create (output|parent) directory"):
                export_lora(model=mock_peft_model, output_dir=temp_dir / "locked")

    def test_export_merged_permission_error(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_merged should raise MergeExportError when output dir can't be created."""
        from backpropagate.exceptions import MergeExportError
        from backpropagate.export import export_merged

        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(MergeExportError, match="Cannot create (output|parent) directory"):
                export_merged(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "locked",
                )

    def test_export_gguf_permission_error(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should raise GGUFExportError when output dir can't be created."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            with pytest.raises(GGUFExportError, match="Cannot create output directory"):
                export_gguf(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "locked",
                    quantization="q4_k_m",
                )


# =============================================================================
# TQ-002: subprocess.TimeoutExpired
# =============================================================================


class TestSubprocessTimeout:
    """Tests for subprocess.TimeoutExpired handling (TQ-002)."""

    def test_export_gguf_llama_cpp_timeout(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should raise GGUFExportError when llama.cpp conversion times out.

        BRIDGE-A-004 (v1.1.2 amend wave): the timeout path now flows
        through ``_run_subprocess_interruptible`` (Popen-based), not
        ``subprocess.run``. The mock target moved accordingly. The helper
        re-raises ``subprocess.TimeoutExpired`` after killing the child
        process group, which the export_gguf except branch translates
        into a GGUFExportError with the "timed out" message.
        """
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        with patch("backpropagate.export._has_unsloth", return_value=False), \
             patch("backpropagate.export._is_peft_model", return_value=True), \
             patch.object(Path, "exists", return_value=True), \
             patch(
                 "backpropagate.export._run_subprocess_interruptible",
                 side_effect=subprocess.TimeoutExpired(cmd="python convert", timeout=1800),
             ):
            with pytest.raises(GGUFExportError, match="timed out"):
                export_gguf(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "gguf_timeout",
                    quantization="q4_k_m",
                )

    def test_register_with_ollama_timeout(self, sample_gguf_path):
        """register_with_ollama should raise OllamaRegistrationError on timeout.

        BRIDGE-A-004 (v1.1.2 amend wave): like the llama.cpp test above,
        the timeout path is now via ``_run_subprocess_interruptible``. The
        mock target moved from ``subprocess.run`` to the helper so the
        synthesized TimeoutExpired actually reaches the
        OllamaRegistrationError translation branch.
        """
        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import register_with_ollama

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch(
                 "backpropagate.export._run_subprocess_interruptible",
                 side_effect=subprocess.TimeoutExpired(cmd="ollama create", timeout=600),
             ):
            with pytest.raises(OllamaRegistrationError, match="timed out"):
                register_with_ollama(sample_gguf_path, "test-model")


# =============================================================================
# TQ-003: Empty GGUF file (0 bytes)
# =============================================================================


class TestEmptyGgufFile:
    """Tests for empty GGUF file detection (TQ-003)."""

    def test_export_gguf_empty_file_raises(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should raise GGUFExportError when output file is 0 bytes."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        # Unsloth save creates an empty GGUF file (0 bytes)
        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model-q4_k_m.gguf").write_bytes(b"")  # Empty file

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        with patch("backpropagate.export._has_unsloth", return_value=True):
            with pytest.raises(GGUFExportError, match="empty.*0 bytes"):
                export_gguf(
                    model=mock_peft_model,
                    tokenizer=mock_tokenizer,
                    output_dir=temp_dir / "gguf_empty",
                    quantization="q4_k_m",
                )


# =============================================================================
# F-004 MODEL CARD EMISSION TESTS
# =============================================================================

class TestExportLoraModelCard:
    """Verify export_lora writes model_card.md when enabled."""

    def test_emits_model_card_by_default(self, temp_dir):
        from backpropagate.export import export_lora

        # Create source adapter files
        src_dir = temp_dir / "source"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{"test": true}')
        (src_dir / "adapter_model.safetensors").write_bytes(b"mock")

        output_dir = temp_dir / "output"
        export_lora(model=src_dir, output_dir=output_dir)

        card_path = output_dir / "model_card.md"
        assert card_path.exists(), "Default emit_model_card=True must write a card"
        content = card_path.read_text(encoding="utf-8")
        assert "library_name: backpropagate" in content
        assert "Trust signals" in content

    def test_no_model_card_opts_out(self, temp_dir):
        from backpropagate.export import export_lora

        src_dir = temp_dir / "source"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{"test": true}')
        (src_dir / "adapter_model.safetensors").write_bytes(b"mock")

        output_dir = temp_dir / "output"
        export_lora(
            model=src_dir,
            output_dir=output_dir,
            emit_model_card=False,
        )

        assert not (output_dir / "model_card.md").exists()

    def test_model_card_pulls_run_history(self, temp_dir):
        from backpropagate.checkpoints import RunHistoryManager
        from backpropagate.export import export_lora

        # Stage a RunHistoryManager record in the output root.
        output_root = temp_dir / "output"
        output_root.mkdir()
        manager = RunHistoryManager(str(output_root))
        manager.record_run_started(
            run_id="testrun123",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dataset_info="data.jsonl",
            hyperparameters={"lora_r": 32, "seed": 42},
        )
        manager.record_run_completed(
            run_id="testrun123",
            final_loss=0.125,
            loss_history=[1.0, 0.5, 0.125],
            steps=200,
        )

        # Source adapter.
        src_dir = temp_dir / "source"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{}')
        (src_dir / "adapter_model.safetensors").write_bytes(b"x")

        export_dir = output_root / "lora"
        export_lora(
            model=src_dir,
            output_dir=export_dir,
            run_id="testrun123",
            output_root=output_root,
        )

        card = (export_dir / "model_card.md").read_text(encoding="utf-8")
        assert "testrun123" in card
        assert "Qwen/Qwen2.5-7B-Instruct" in card
        assert "0.1250" in card  # final loss
        assert "| Steps | 200 |" in card
        assert "| LoRA rank | 32 |" in card

    def test_model_card_incomplete_provenance_without_history(self, temp_dir):
        from backpropagate.export import export_lora

        src_dir = temp_dir / "source"
        src_dir.mkdir()
        (src_dir / "adapter_config.json").write_text('{}')
        (src_dir / "adapter_model.safetensors").write_bytes(b"x")

        output_dir = temp_dir / "noprov"
        export_lora(
            model=src_dir,
            output_dir=output_dir,
            run_id="orphan",
            base_model="Qwen/Qwen2.5-7B-Instruct",
        )

        card = (output_dir / "model_card.md").read_text(encoding="utf-8")
        assert "Incomplete provenance" in card
        # Base model still surfaces because we passed it explicitly.
        assert "Qwen/Qwen2.5-7B-Instruct" in card


class TestExportMergedModelCard:
    """Verify export_merged writes model_card.md."""

    def test_emits_model_card_after_merge(self, temp_dir, mock_peft_model, mock_tokenizer):
        from backpropagate.export import export_merged

        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        output_dir = temp_dir / "merged"
        with patch("backpropagate.export._is_peft_model", return_value=True):
            export_merged(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=output_dir,
                base_model="Qwen/Qwen2.5-7B-Instruct",
            )

        card_path = output_dir / "model_card.md"
        assert card_path.exists()
        assert "Qwen/Qwen2.5-7B-Instruct" in card_path.read_text(encoding="utf-8")

    def test_no_model_card_opts_out(self, temp_dir, mock_peft_model, mock_tokenizer):
        from backpropagate.export import export_merged

        merged_model = MagicMock()
        merged_model.save_pretrained = MagicMock()
        mock_peft_model.merge_and_unload.return_value = merged_model

        output_dir = temp_dir / "merged_no_card"
        with patch("backpropagate.export._is_peft_model", return_value=True):
            export_merged(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=output_dir,
                emit_model_card=False,
            )

        assert not (output_dir / "model_card.md").exists()


class TestExportGgufModelCard:
    """Verify export_gguf writes model_card.md next to the GGUF file."""

    def test_emits_model_card_with_quantization_tag(self, temp_dir, mock_peft_model, mock_tokenizer):
        from backpropagate.export import export_gguf

        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model-q4_k_m.gguf").write_bytes(b"x" * 1024)

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        output_dir = temp_dir / "gguf"
        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=output_dir,
                quantization="q4_k_m",
            )

        # The card lives in the same dir as the GGUF file.
        card_path = result.path.parent / "model_card.md"
        assert card_path.exists()
        content = card_path.read_text(encoding="utf-8")
        assert "q4_k_m" in content
        assert "  - gguf" in content

    def test_no_model_card_opts_out(self, temp_dir, mock_peft_model, mock_tokenizer):
        from backpropagate.export import export_gguf

        def mock_save_gguf(path, tokenizer, quantization_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model-q4_k_m.gguf").write_bytes(b"x" * 1024)

        mock_peft_model.save_pretrained_gguf = mock_save_gguf

        output_dir = temp_dir / "gguf_no_card"
        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=output_dir,
                quantization="q4_k_m",
                emit_model_card=False,
            )

        card_path = result.path.parent / "model_card.md"
        assert not card_path.exists()


# =============================================================================
# F-001 HUB PUSH TESTS
# =============================================================================

class TestPushToHub:
    """push_to_hub uploads + error wrapping."""

    def test_missing_local_path_raises_export_error(self, tmp_path):
        from backpropagate.exceptions import ExportError
        from backpropagate.export import push_to_hub

        with pytest.raises(ExportError, match="does not exist"):
            push_to_hub(
                local_path=tmp_path / "missing",
                repo_id="alice/missing",
            )

    def test_directory_upload_calls_upload_folder(self, tmp_path):
        from backpropagate.export import push_to_hub

        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        (local_dir / "adapter_config.json").write_text("{}")
        (local_dir / "adapter_model.safetensors").write_bytes(b"x")
        (local_dir / "model_card.md").write_text("# Model card\n")

        fake_api = MagicMock()
        fake_api.upload_folder = MagicMock()
        fake_create_repo = MagicMock()
        fake_hf_module = MagicMock()
        fake_hf_module.HfApi = MagicMock(return_value=fake_api)
        fake_hf_module.create_repo = fake_create_repo
        fake_utils = MagicMock()

        class _DummyHTTPError(Exception):
            pass

        fake_utils.HfHubHTTPError = _DummyHTTPError

        import sys

        with patch.dict(sys.modules, {
            "huggingface_hub": fake_hf_module,
            "huggingface_hub.utils": fake_utils,
        }):
            url = push_to_hub(
                local_path=local_dir,
                repo_id="alice/adapter",
                token="hf_test",
            )

        assert url == "https://huggingface.co/alice/adapter"
        fake_create_repo.assert_called_once()
        fake_api.upload_folder.assert_called_once()
        kwargs = fake_api.upload_folder.call_args.kwargs
        assert kwargs["repo_id"] == "alice/adapter"
        # The model card was mirrored to README.md before upload.
        assert (local_dir / "README.md").exists()

    def test_single_file_upload(self, tmp_path):
        from backpropagate.export import push_to_hub

        local_file = tmp_path / "model.gguf"
        local_file.write_bytes(b"x" * 1024)

        fake_api = MagicMock()
        fake_hf_module = MagicMock()
        fake_hf_module.HfApi = MagicMock(return_value=fake_api)
        fake_hf_module.create_repo = MagicMock()
        fake_utils = MagicMock()

        class _DummyHTTPError(Exception):
            pass

        fake_utils.HfHubHTTPError = _DummyHTTPError

        import sys

        with patch.dict(sys.modules, {
            "huggingface_hub": fake_hf_module,
            "huggingface_hub.utils": fake_utils,
        }):
            push_to_hub(
                local_path=local_file,
                repo_id="alice/gguf",
            )

        fake_api.upload_file.assert_called_once()
        fake_api.upload_folder.assert_not_called()

    def test_auth_401_wrapped_with_auth_code(self, tmp_path):
        from backpropagate.exceptions import ExportError
        from backpropagate.export import push_to_hub

        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        (local_dir / "adapter_config.json").write_text("{}")

        class _Resp:
            status_code = 401

        class _Err(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                self.response = _Resp()

        fake_api = MagicMock()
        fake_api.upload_folder = MagicMock(side_effect=_Err("unauthorized"))
        fake_hf_module = MagicMock()
        fake_hf_module.HfApi = MagicMock(return_value=fake_api)
        fake_hf_module.create_repo = MagicMock()
        fake_utils = MagicMock()
        fake_utils.HfHubHTTPError = _Err

        import sys

        with patch.dict(sys.modules, {
            "huggingface_hub": fake_hf_module,
            "huggingface_hub.utils": fake_utils,
        }):
            with pytest.raises(ExportError) as excinfo:
                push_to_hub(
                    local_path=local_dir,
                    repo_id="alice/adapter",
                )

        # BRIDGE-A-007 (v1.1.2 amend wave): the 401/403 branch now sets
        # the canonical INPUT_AUTH_REQUIRED code (already in ERROR_CODES)
        # instead of the orphaned HUB_PUSH_AUTH string. cmd_push accepts
        # both during the rename, but new errors emit INPUT_AUTH_REQUIRED.
        assert getattr(excinfo.value, "code", None) == "INPUT_AUTH_REQUIRED"

    def test_huggingface_hub_missing_emits_clear_export_error(self, tmp_path):
        from backpropagate.exceptions import ExportError
        from backpropagate.export import push_to_hub

        local_dir = tmp_path / "adapter"
        local_dir.mkdir()
        (local_dir / "adapter_config.json").write_text("{}")

        # Simulate `huggingface_hub` missing by intercepting the import.
        import builtins

        real_import = builtins.__import__

        def _import_blocker(name, *args, **kwargs):
            if name.startswith("huggingface_hub"):
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_import_blocker):
            with pytest.raises(ExportError, match="huggingface_hub is not installed"):
                push_to_hub(local_path=local_dir, repo_id="alice/adapter")

    def test_resolve_hf_token_prefers_explicit(self, tmp_path, monkeypatch):
        from backpropagate.export import _resolve_hf_token

        monkeypatch.setenv("HF_TOKEN", "env_token")
        assert _resolve_hf_token("explicit") == "explicit"

    def test_resolve_hf_token_env_fallback(self, tmp_path, monkeypatch):
        from backpropagate.export import _resolve_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf-env")

        # Avoid picking up a real ~/.cache/huggingface/token on this machine
        # — patch Path.home to a clean tmp dir.
        with patch("backpropagate.export.Path.home", return_value=tmp_path):
            assert _resolve_hf_token(None) == "hf-env"

    def test_resolve_hf_token_returns_none_when_unset(self, tmp_path, monkeypatch):
        from backpropagate.export import _resolve_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

        with patch("backpropagate.export.Path.home", return_value=tmp_path):
            assert _resolve_hf_token(None) is None


# =============================================================================
# ATOMIC EXPORT WRITES (B-006) — TESTS-A-005
# =============================================================================
#
# CHANGELOG.md L32 documents the atomic-write contract for export_lora and
# export_gguf: both write into ``<path>.partial`` first and ``shutil.move`` to
# the final path on success. A mid-write failure (disk full, OOM, etc.) MUST
# leave neither a partial-named directory nor a half-written final artifact.
#
# These tests pin the contract by making the inner write step raise during
# the partial stage and asserting:
#   (a) the FINAL target path does not exist after the failure
#   (b) the ``.partial`` sibling is cleaned up
#
# Pairs with TestTrainerSaveAtomic + TestSLAOMergerSaveAtomic in test_trainer.py.


class TestExportLoraAtomicWrite:
    """Pin the atomic-write contract for export_lora (B-006, TESTS-A-005)."""

    def test_export_lora_happy_path(self, temp_dir, mock_peft_model):
        """export_lora: success path promotes partial to final, no residue."""
        from backpropagate.export import export_lora

        target = temp_dir / "lora_export"

        # Make save_pretrained drop a marker file so we can prove the move ran.
        def write_marker(path, *args, **kwargs):
            from pathlib import Path
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")

        mock_peft_model.save_pretrained.side_effect = write_marker

        with patch("backpropagate.export._is_peft_model", return_value=True):
            result = export_lora(
                model=mock_peft_model,
                output_dir=target,
                emit_model_card=False,
            )

        assert target.exists(), "Final export dir must exist on success"
        assert (target / "adapter_model.safetensors").exists(), (
            "Marker written in partial stage must be promoted to final dir"
        )
        partial = target.with_name(target.name + ".partial")
        assert not partial.exists(), (
            f"Partial dir must be cleaned up on success; still at {partial}"
        )
        assert result.path == target

    def test_export_lora_disk_full_leaves_no_final_artifact(self, temp_dir,
                                                             mock_peft_model):
        """export_lora: a mid-write OSError must NOT leave a half-written final dir."""
        from backpropagate.exceptions import ExportError
        from backpropagate.export import export_lora

        target = temp_dir / "lora_export"

        # Simulate disk-full: save_pretrained raises mid-write. The partial
        # directory has already been created at this point.
        mock_peft_model.save_pretrained.side_effect = OSError(
            "[Errno 28] No space left on device"
        )

        with patch("backpropagate.export._is_peft_model", return_value=True), \
             pytest.raises(ExportError):
            export_lora(
                model=mock_peft_model,
                output_dir=target,
                emit_model_card=False,
            )

        # The atomic contract: no final artifact, no partial residue.
        assert not target.exists(), (
            "Final export dir must NOT exist after a mid-write failure"
        )
        partial = target.with_name(target.name + ".partial")
        assert not partial.exists(), (
            f"Partial dir must be cleaned up on failure; still at {partial}"
        )


class TestExportGgufAtomicWrite:
    """Pin the atomic-write contract for export_gguf (B-006, TESTS-A-005).

    export_gguf writes via Unsloth into ``output_dir / _unsloth_partial`` then
    moves the produced .gguf into ``output_dir`` proper. A mid-conversion
    failure must clean up the partial scratch dir and leave no .gguf at the
    final path.
    """

    def test_export_gguf_happy_path(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf: Unsloth success path produces final .gguf, no scratch left."""
        from backpropagate.export import export_gguf

        def fake_save_gguf(path, tokenizer, quantization_method):
            from pathlib import Path
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            # Unsloth writes the .gguf into the partial dir; export_gguf
            # moves it up to output_path on success.
            (p / f"model-{quantization_method}.gguf").write_bytes(b"GGUF" * 256)

        mock_peft_model.save_pretrained_gguf = fake_save_gguf

        target = temp_dir / "gguf_export"

        with patch("backpropagate.export._has_unsloth", return_value=True):
            result = export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=target,
                quantization="q4_k_m",
                emit_model_card=False,
            )

        # The produced .gguf must be at the final path (not in the partial).
        final_gguf = target / "model-q4_k_m.gguf"
        assert final_gguf.exists(), (
            f"Promoted .gguf must exist at final path; expected {final_gguf}"
        )
        scratch = target / "_unsloth_partial"
        assert not scratch.exists(), (
            f"Unsloth scratch dir must be cleaned up on success; still at {scratch}"
        )
        assert result.path == final_gguf

    def test_export_gguf_disk_full_leaves_no_final_gguf(self, temp_dir,
                                                         mock_peft_model,
                                                         mock_tokenizer):
        """export_gguf: Unsloth raising mid-write must NOT leave a final .gguf."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_gguf

        # Simulate disk-full during the Unsloth save step.
        def boom(path, tokenizer, quantization_method):
            raise OSError("[Errno 28] No space left on device")

        mock_peft_model.save_pretrained_gguf = boom

        target = temp_dir / "gguf_export"

        with patch("backpropagate.export._has_unsloth", return_value=True), \
             pytest.raises((GGUFExportError, OSError)):
            export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=target,
                quantization="q4_k_m",
                emit_model_card=False,
            )

        # No final .gguf at the canonical path.
        final_gguf = target / "model-q4_k_m.gguf"
        assert not final_gguf.exists(), (
            f"Final .gguf must NOT exist after disk-full mid-write; "
            f"unexpectedly found at {final_gguf}"
        )
        # Scratch dir cleaned up.
        scratch = target / "_unsloth_partial"
        assert not scratch.exists(), (
            f"Unsloth scratch dir must be cleaned up on failure; still at {scratch}"
        )


# =============================================================================
# STAGE C PROACTIVE AMEND — DATA-B-004/005/007/008
# =============================================================================


class _FakeParam:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _FakeModel:
    """Minimal stand-in exposing parameters() like a torch Module."""

    def __init__(self, n_params):
        self._params = [_FakeParam(n_params)]

    def parameters(self):
        return list(self._params)


class TestStageCDiskSpaceGuard:
    """DATA-B-004: pre-flight free-space check before a multi-GB export."""

    def test_estimate_param_count(self):
        from backpropagate.export import _estimate_param_count

        assert _estimate_param_count(_FakeModel(7_000_000_000)) == 7_000_000_000

    def test_estimate_param_count_non_model_returns_none(self):
        from backpropagate.export import _estimate_param_count

        class NoParams:
            pass

        assert _estimate_param_count(NoParams()) is None

    def test_disk_guard_raises_when_short(self, temp_dir):
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import _check_export_disk_space

        # Absurd multiplier forces "insufficient space" regardless of the
        # real free space on the test volume.
        with pytest.raises(GGUFExportError) as exc:
            _check_export_disk_space(
                Path(temp_dir),
                _FakeModel(7_000_000_000),
                error_cls=GGUFExportError,
                multiplier=1_000_000.0,
                output_path=str(temp_dir),
                quantization="q4_k_m",
            )
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"
        assert "disk space" in str(exc.value).lower()
        # Error carries the export-detail kwargs through.
        assert exc.value.quantization == "q4_k_m"

    def test_disk_guard_merge_error_class(self, temp_dir):
        from backpropagate.exceptions import MergeExportError
        from backpropagate.export import _check_export_disk_space

        with pytest.raises(MergeExportError) as exc:
            _check_export_disk_space(
                Path(temp_dir),
                _FakeModel(7_000_000_000),
                error_cls=MergeExportError,
                multiplier=1_000_000.0,
            )
        assert exc.value.code == "RUNTIME_MERGE_EXPORT_FAILED"

    def test_disk_guard_noop_when_params_unknown(self, temp_dir):
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import _check_export_disk_space

        class NoParams:
            pass

        # Must NOT raise even with an absurd multiplier — unknown param count
        # means the guard cannot make a sound estimate, so it stands down.
        _check_export_disk_space(
            Path(temp_dir),
            NoParams(),
            error_cls=GGUFExportError,
            multiplier=1_000_000.0,
        )

    def test_disk_guard_passes_when_space_available(self, temp_dir):
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import _check_export_disk_space

        # 1KB model with default multiplier needs a few KB — always available.
        _check_export_disk_space(
            Path(temp_dir),
            _FakeModel(1_000),
            error_cls=GGUFExportError,
        )


class TestFP8MergedBf16Cast:
    """FP8 finding: an FP8-trained model merges to an f32 checkpoint (torchao
    Float8Linear upcasts its weight to f32). At 4 B/param that doubles the
    on-disk size the disk pre-flight budgeted (2 B/param), so the merged/GGUF
    export paths down-cast the merge to bf16 BEFORE save_pretrained. These
    tests pin that contract with tiny stub models — no real FP8 run needed.
    """

    def _f32_stub_model(self):
        """A stub model whose params report torch.float32 and whose ``.to()``
        records the requested dtype (mimicking the post-merge FP8 model)."""
        import torch

        class _F32Stub:
            def __init__(self):
                self._dtype = torch.float32
                self.to_calls: list = []
                self.save_pretrained = MagicMock()

            def parameters(self):
                # One real f32 tensor so _model_has_float32_params sees f32.
                return [torch.zeros(2, 2, dtype=self._dtype)]

            def to(self, dtype):
                self.to_calls.append(dtype)
                self._dtype = dtype
                return self  # Module.to mutates in place and returns self

        return _F32Stub()

    def test_helper_casts_f32_model_to_bf16(self):
        """``_cast_merged_model_to_bf16`` invokes ``.to(torch.bfloat16)`` on a
        model carrying f32 params and the result reports bf16 afterward."""
        import torch

        from backpropagate.export import _cast_merged_model_to_bf16

        stub = self._f32_stub_model()
        result = _cast_merged_model_to_bf16(stub)
        assert torch.bfloat16 in stub.to_calls, (
            "FP8 finding: an f32 merged model must be cast via .to(bfloat16)."
        )
        # The returned model's params now report bf16.
        assert all(p.dtype == torch.bfloat16 for p in result.parameters())

    def test_helper_noop_on_bf16_model(self):
        """A model already in bf16 (the normal LoRA merge) must NOT be cast —
        no spurious .to() call, returned unchanged."""
        import torch

        from backpropagate.export import _cast_merged_model_to_bf16

        class _Bf16Stub:
            def __init__(self):
                self.to_calls: list = []

            def parameters(self):
                return [torch.zeros(2, 2, dtype=torch.bfloat16)]

            def to(self, dtype):
                self.to_calls.append(dtype)
                return self

        stub = _Bf16Stub()
        result = _cast_merged_model_to_bf16(stub)
        assert stub.to_calls == [], (
            "FP8 finding: a bf16 merge must not be re-cast (no .to() call)."
        )
        assert result is stub

    def test_helper_noop_on_stub_without_dtypes(self):
        """A non-torch / mock model (no real dtypes) is returned unchanged and
        never raises — the cast is best-effort."""
        from backpropagate.export import _cast_merged_model_to_bf16

        class _NoParams:
            pass

        sentinel = _NoParams()
        assert _cast_merged_model_to_bf16(sentinel) is sentinel

    def test_export_merged_casts_f32_merge_to_bf16(self, temp_dir, mock_tokenizer):
        """End-to-end: export_merged of a PEFT model whose merge yields an
        f32 model down-casts to bf16 before save_pretrained, so the saved
        checkpoint matches the disk-guard's 2 B/param budget."""
        import torch

        from backpropagate.export import export_merged

        merged_f32 = self._f32_stub_model()
        mock_peft = MagicMock()
        mock_peft.merge_and_unload.return_value = merged_f32

        with patch("backpropagate.export._is_peft_model", return_value=True):
            export_merged(
                model=mock_peft,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir / "merged_fp8",
                emit_model_card=False,
            )

        assert torch.bfloat16 in merged_f32.to_calls, (
            "FP8 finding: export_merged must down-cast an f32 merge to bf16 "
            "before saving."
        )
        merged_f32.save_pretrained.assert_called_once()


class TestStageCListOllamaWarns:
    """DATA-B-005: list_ollama_models WARNs (naming the cause) on a
    missing CLI / daemon-down / timeout instead of silently returning []."""

    def test_warns_when_cli_missing(self, caplog):
        import logging

        from backpropagate.export import list_ollama_models

        with patch("shutil.which", return_value=None):
            with caplog.at_level(logging.WARNING, logger="backpropagate.export"):
                assert list_ollama_models() == []
        assert any("not on PATH" in r.message for r in caplog.records)

    def test_warns_on_daemon_error(self, caplog):
        import logging

        from backpropagate.export import list_ollama_models

        err = subprocess.CalledProcessError(1, "ollama")
        err.stderr = "connection refused"
        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", side_effect=err):
            with caplog.at_level(logging.WARNING, logger="backpropagate.export"):
                assert list_ollama_models() == []
        joined = " ".join(r.message for r in caplog.records)
        assert "ollama serve" in joined

    def test_warns_on_timeout(self, caplog):
        import logging

        from backpropagate.export import list_ollama_models

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ollama", 30)):
            with caplog.at_level(logging.WARNING, logger="backpropagate.export"):
                assert list_ollama_models() == []
        assert any("timed out" in r.message for r in caplog.records)


class TestStageCHfTokenCap:
    """DATA-B-007: cached HF token read is size-capped + single-line."""

    def test_oversized_token_file_is_capped(self, temp_dir, monkeypatch):
        from backpropagate import export

        home = Path(temp_dir)
        cache = home / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        # First line is a valid-looking token; then a huge junk tail.
        (cache / "token").write_text("hf_realtoken\n" + ("X" * 1_000_000), encoding="utf-8")
        monkeypatch.setattr(export.Path, "home", staticmethod(lambda: home))

        token = export._resolve_hf_token(None)
        assert token == "hf_realtoken"  # only the first line, capped read

    def test_no_token_file_returns_none(self, temp_dir, monkeypatch):
        from backpropagate import export

        monkeypatch.setattr(export.Path, "home", staticmethod(lambda: Path(temp_dir)))
        # No env token, no cache file.
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert export._resolve_hf_token(None) is None


class TestStageCExportProgressLogged:
    """DATA-B-008: the long GGUF step logs (not just print) so headless runs
    record that it started."""

    def test_gguf_progress_is_logged(self, temp_dir, caplog):
        import logging

        from backpropagate import export

        model = MagicMock()
        tokenizer = MagicMock()

        # save_pretrained_gguf writes a .gguf into the partial dir.
        def fake_save(partial_dir, tok, quantization_method):
            p = Path(partial_dir) / "model.gguf"
            p.write_bytes(b"GGUF" + b"\x00" * 64)

        model.save_pretrained_gguf.side_effect = fake_save

        with patch("backpropagate.export._has_unsloth", return_value=True), \
             patch("backpropagate.export._is_peft_model", return_value=False), \
             patch("backpropagate.export._estimate_param_count", return_value=None), \
             caplog.at_level(logging.INFO, logger="backpropagate.export"):
            export.export_gguf(
                model,
                tokenizer,
                str(temp_dir),
                quantization="q4_k_m",
                emit_model_card=False,
            )

        assert any("Exporting to GGUF" in r.message for r in caplog.records)


# =============================================================================
# v1.5 T2.3 — ADAPTER-NATIVE OLLAMA EXPORT ("ollama-adapter") + ADAPTER SHELF
# =============================================================================
#
# V1_5_BRIEF §2 findings 27–29 / §3 T2.3. We implement the SAFETENSORS
# ``ADAPTER`` mechanism (Ollama loads a safetensors LoRA adapter directory
# directly via the ADAPTER Modelfile directive — no GGUF-adapter conversion).
# These tests mock the filesystem (a fake safetensors adapter dir) and the
# ollama subprocess; they never require a real Ollama daemon.


def _make_safetensors_adapter(root: Path, name: str = "my-adapter") -> Path:
    """Create a minimal safetensors LoRA adapter directory (export_lora shape)."""
    adapter_dir = root / name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"mock safetensors")
    return adapter_dir


class TestCreateAdapterModelfile:
    """create_adapter_modelfile emits FROM <base> + ADAPTER <path>."""

    def test_emits_from_and_adapter_directives(self, temp_dir):
        from backpropagate.export import create_adapter_modelfile

        adapter_dir = _make_safetensors_adapter(temp_dir)

        modelfile = create_adapter_modelfile("llama3.2", adapter_dir)

        assert modelfile.exists()
        content = modelfile.read_text(encoding="utf-8")
        # FROM names the BASE (bare Ollama name -> unquoted).
        assert "FROM llama3.2" in content
        # ADAPTER points at the resolved adapter dir (escaped FROM-style).
        expected_adapter = (
            str(adapter_dir.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        )
        assert f'ADAPTER "{expected_adapter}"' in content
        # No merge happened: there is no merged GGUF FROM line.
        assert "ADAPTER" in content

    def test_default_modelfile_path_is_in_adapter_dir(self, temp_dir):
        from backpropagate.export import create_adapter_modelfile

        adapter_dir = _make_safetensors_adapter(temp_dir)
        modelfile = create_adapter_modelfile("llama3.2", adapter_dir)
        assert modelfile.name == "Modelfile"
        assert modelfile.parent == adapter_dir.resolve()

    def test_base_model_path_is_quoted(self, temp_dir):
        """A base given as a path (drive colon / separator) is quoted in FROM."""
        from backpropagate.export import create_adapter_modelfile

        adapter_dir = _make_safetensors_adapter(temp_dir)
        base_path = str(temp_dir / "base.gguf")
        modelfile = create_adapter_modelfile(base_path, adapter_dir)
        content = modelfile.read_text(encoding="utf-8")
        escaped_base = base_path.replace("\\", "\\\\").replace('"', '\\"')
        assert f'FROM "{escaped_base}"' in content

    def test_options_baked_in(self, temp_dir):
        from backpropagate.export import create_adapter_modelfile

        adapter_dir = _make_safetensors_adapter(temp_dir)
        modelfile = create_adapter_modelfile(
            "llama3.2",
            adapter_dir,
            system_prompt="You are helpful.",
            temperature=0.9,
            context_length=8192,
        )
        content = modelfile.read_text(encoding="utf-8")
        assert "0.9" in content
        assert "8192" in content
        assert "You are helpful." in content

    def test_accepts_gguf_adapter_file(self, temp_dir):
        """A .gguf adapter file is accepted directly by ADAPTER."""
        from backpropagate.export import create_adapter_modelfile

        gguf_adapter = temp_dir / "ollama-lora.gguf"
        gguf_adapter.write_bytes(b"GGUF adapter")
        modelfile = create_adapter_modelfile(
            "llama3.2", gguf_adapter, output_path=temp_dir / "Modelfile"
        )
        content = modelfile.read_text(encoding="utf-8")
        expected = (
            str(gguf_adapter.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        )
        assert f'ADAPTER "{expected}"' in content

    def test_rejects_control_char_in_base(self, temp_dir):
        from backpropagate.exceptions import ExportError
        from backpropagate.export import create_adapter_modelfile

        adapter_dir = _make_safetensors_adapter(temp_dir)
        with pytest.raises(ExportError) as exc:
            create_adapter_modelfile("llama\n3.2", adapter_dir)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_rejects_control_char_in_system_prompt(self, temp_dir):
        from backpropagate.exceptions import ExportError
        from backpropagate.export import create_adapter_modelfile

        adapter_dir = _make_safetensors_adapter(temp_dir)
        with pytest.raises(ExportError) as exc:
            create_adapter_modelfile(
                "llama3.2", adapter_dir, system_prompt="bad\x00prompt"
            )
        assert exc.value.code == "INPUT_VALIDATION_FAILED"


class TestExportOllamaAdapterDegrade:
    """The degrade path: no Ollama-loadable adapter -> structured GGUFExportError."""

    def test_bin_only_adapter_degrades_with_actionable_hint(self, temp_dir):
        """A .bin-only adapter dir (no safetensors, no gguf) degrades cleanly."""
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_ollama_adapter

        adapter_dir = temp_dir / "legacy"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.bin").write_bytes(b"legacy bin")

        with pytest.raises(GGUFExportError) as exc:
            export_ollama_adapter(adapter_dir, base_model="llama3.2")
        # Reuses the EXISTING GGUF export code — no new code added.
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"
        # Names the fix: re-export safetensors OR convert_lora_to_gguf.py.
        assert "convert_lora_to_gguf.py" in exc.value.suggestion
        assert "--format lora" in exc.value.suggestion

    def test_empty_dir_degrades(self, temp_dir):
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_ollama_adapter

        empty = temp_dir / "empty"
        empty.mkdir()
        with pytest.raises(GGUFExportError) as exc:
            export_ollama_adapter(empty, base_model="llama3.2")
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"

    def test_missing_path_degrades(self, temp_dir):
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_ollama_adapter

        with pytest.raises(GGUFExportError) as exc:
            export_ollama_adapter(temp_dir / "nope", base_model="llama3.2")
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"

    def test_unrecognised_file_suffix_degrades(self, temp_dir):
        from backpropagate.exceptions import GGUFExportError
        from backpropagate.export import export_ollama_adapter

        bad = temp_dir / "adapter.bin"
        bad.write_bytes(b"x")
        with pytest.raises(GGUFExportError) as exc:
            export_ollama_adapter(bad, base_model="llama3.2")
        assert exc.value.code == "RUNTIME_GGUF_EXPORT_FAILED"


class TestExportOllamaAdapterModelfileOnly:
    """modelfile_only=True writes the Modelfile without invoking ollama."""

    def test_modelfile_only_no_subprocess(self, temp_dir):
        from backpropagate.export import ExportFormat, export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir, name="taskA")

        # If ANY subprocess fires, fail loudly.
        with patch(
            "backpropagate.export._run_subprocess_interruptible",
            side_effect=AssertionError("ollama must NOT be invoked when modelfile_only=True"),
        ), patch("subprocess.run", side_effect=AssertionError("no subprocess allowed")):
            result = export_ollama_adapter(
                adapter_dir, base_model="llama3.2", modelfile_only=True
            )

        assert result.format == ExportFormat.OLLAMA_ADAPTER
        # The Modelfile is written and kept.
        assert result.path.exists()
        content = result.path.read_text(encoding="utf-8")
        assert "FROM llama3.2" in content
        assert "ADAPTER" in content
        # Derived model name recorded in the quantization slot.
        assert result.quantization == "llama3.2:taskA"

    def test_modelfile_only_does_not_require_ollama_on_path(self, temp_dir):
        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir)
        # Even with no ollama CLI, modelfile_only must succeed.
        with patch("shutil.which", return_value=None):
            result = export_ollama_adapter(
                adapter_dir, base_model="llama3.2", modelfile_only=True
            )
        assert result.path.exists()


class TestExportOllamaAdapterRegistration:
    """The registration path mocks `ollama create` and asserts the argv."""

    def test_registration_calls_ollama_create_with_right_args(self, temp_dir):
        import subprocess as _sp

        from backpropagate.export import ExportFormat, export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir, name="taskB")

        mock_result = _sp.CompletedProcess(
            args=["ollama", "create"], returncode=0, stdout="", stderr=""
        )

        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "backpropagate.export._run_subprocess_interruptible",
            return_value=mock_result,
        ) as mock_run:
            result = export_ollama_adapter(adapter_dir, base_model="llama3.2")

        assert result.format == ExportFormat.OLLAMA_ADAPTER
        mock_run.assert_called_once()
        argv = mock_run.call_args.args[0]
        # ['ollama', 'create', '<base>:<tag>', '-f', '<Modelfile>']
        assert argv[0] == "ollama"
        assert argv[1] == "create"
        assert argv[2] == "llama3.2:taskB"
        assert argv[3] == "-f"
        # The Modelfile arg is a real path string.
        assert argv[4].endswith("Modelfile")
        # Default behaviour cleans up the Modelfile after a successful create.
        assert not result.path.exists()

    def test_registration_keep_modelfile(self, temp_dir):
        import subprocess as _sp

        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir)
        mock_result = _sp.CompletedProcess(
            args=["ollama", "create"], returncode=0, stdout="", stderr=""
        )
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "backpropagate.export._run_subprocess_interruptible",
            return_value=mock_result,
        ):
            result = export_ollama_adapter(
                adapter_dir, base_model="llama3.2", keep_modelfile=True
            )
        assert result.path.exists()

    def test_base_tag_strips_base_model_tag(self, temp_dir):
        """base_model='mistral:7b' -> model name 'mistral:<adapter-tag>'."""
        import subprocess as _sp

        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir, name="custom")
        mock_result = _sp.CompletedProcess(
            args=["ollama", "create"], returncode=0, stdout="", stderr=""
        )
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "backpropagate.export._run_subprocess_interruptible",
            return_value=mock_result,
        ) as mock_run:
            export_ollama_adapter(adapter_dir, base_model="mistral:7b")
        argv = mock_run.call_args.args[0]
        # The base's own ':7b' tag is dropped; the adapter tag takes its place.
        assert argv[2] == "mistral:custom"

    def test_registration_failure_raises_ollama_error(self, temp_dir):
        import subprocess

        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir)
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "backpropagate.export._run_subprocess_interruptible",
            side_effect=subprocess.CalledProcessError(1, "ollama"),
        ):
            with pytest.raises(OllamaRegistrationError, match="ollama create failed"):
                export_ollama_adapter(adapter_dir, base_model="llama3.2")

    def test_registration_timeout_raises_ollama_error(self, temp_dir):
        import subprocess

        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir)
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "backpropagate.export._run_subprocess_interruptible",
            side_effect=subprocess.TimeoutExpired(cmd="ollama create", timeout=600),
        ):
            with pytest.raises(OllamaRegistrationError, match="timed out"):
                export_ollama_adapter(adapter_dir, base_model="llama3.2")

    def test_no_ollama_cli_raises_with_manual_hint(self, temp_dir):
        from backpropagate.exceptions import OllamaRegistrationError
        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir)
        with patch("shutil.which", return_value=None):
            with pytest.raises(OllamaRegistrationError) as exc:
                export_ollama_adapter(adapter_dir, base_model="llama3.2")
        # Names the manual register fallback + the Modelfile path.
        assert "ollama create" in exc.value.suggestion
        assert exc.value.code == "DEP_OLLAMA_REGISTRATION_FAILED"


class TestAdapterTagDerivation:
    """tag=None derives from the adapter dir basename (sanitised)."""

    def test_tag_defaults_to_dir_basename(self, temp_dir):
        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir, name="sentiment-v2")
        result = export_ollama_adapter(
            adapter_dir, base_model="llama3.2", modelfile_only=True
        )
        assert result.quantization == "llama3.2:sentiment-v2"

    def test_explicit_tag_overrides(self, temp_dir):
        from backpropagate.export import export_ollama_adapter

        adapter_dir = _make_safetensors_adapter(temp_dir, name="ignored")
        result = export_ollama_adapter(
            adapter_dir, base_model="llama3.2", tag="prod", modelfile_only=True
        )
        assert result.quantization == "llama3.2:prod"

    def test_dirty_basename_is_sanitised(self, temp_dir):
        """A dir name with illegal tag chars is sanitised, never a leading dash."""
        import re as _re

        from backpropagate.export import _derive_adapter_tag

        # _derive_adapter_tag works off the basename of the path object.
        tag = _derive_adapter_tag(temp_dir / "@@bad name!!")
        assert tag and not tag.startswith("-")
        # All chars are in the Ollama tag allowlist.
        assert _re.fullmatch(r"[A-Za-z0-9._-]+", tag)


class TestListAdapterShelf:
    """list_adapter_shelf parses `ollama list` for adapters on a base."""

    def test_lists_adapters_for_base(self):
        from backpropagate.export import list_adapter_shelf

        mock_result = MagicMock()
        mock_result.stdout = (
            "NAME                 ID              SIZE      MODIFIED\n"
            "llama3.2:taskA       abc123          4.7 GB    2 days ago\n"
            "llama3.2:taskB       def456          4.7 GB    1 day ago\n"
            "llama3.2:latest      000000          4.7 GB    3 days ago\n"
            "mistral:other        111111          4.1 GB    5 days ago\n"
        )
        mock_result.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "subprocess.run", return_value=mock_result
        ):
            shelf = list_adapter_shelf("llama3.2")

        tags = {e.tag for e in shelf}
        # Only the llama3.2 adapter VARIANTS — not :latest (the bare base),
        # not the unrelated mistral model.
        assert tags == {"taskA", "taskB"}
        for e in shelf:
            assert e.base == "llama3.2"
            assert e.model_name.startswith("llama3.2:")

    def test_base_model_tag_is_ignored_for_matching(self):
        """Passing 'llama3.2:7b' matches the same shelf as 'llama3.2'."""
        from backpropagate.export import list_adapter_shelf

        mock_result = MagicMock()
        mock_result.stdout = (
            "NAME              ID        SIZE      MODIFIED\n"
            "llama3.2:taskA    abc       4.7 GB    2 days ago\n"
        )
        mock_result.returncode = 0
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "subprocess.run", return_value=mock_result
        ):
            shelf = list_adapter_shelf("llama3.2:7b")
        assert [e.tag for e in shelf] == ["taskA"]

    def test_parses_size_and_modified(self):
        from backpropagate.export import list_adapter_shelf

        mock_result = MagicMock()
        mock_result.stdout = (
            "NAME              ID        SIZE      MODIFIED\n"
            "llama3.2:taskA    abc       4.7 GB    2 days ago\n"
        )
        mock_result.returncode = 0
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "subprocess.run", return_value=mock_result
        ):
            shelf = list_adapter_shelf("llama3.2")
        assert shelf[0].size == "4.7 GB"
        assert "2 days ago" in shelf[0].modified

    def test_empty_when_no_ollama(self, caplog):
        import logging

        from backpropagate.export import list_adapter_shelf

        with patch("shutil.which", return_value=None):
            with caplog.at_level(logging.WARNING, logger="backpropagate.export"):
                shelf = list_adapter_shelf("llama3.2")
        assert shelf == []
        assert any("not on PATH" in r.message for r in caplog.records)

    def test_empty_on_daemon_error(self, caplog):
        import logging

        from backpropagate.export import list_adapter_shelf

        err = subprocess.CalledProcessError(1, "ollama")
        err.stderr = "connection refused"
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "subprocess.run", side_effect=err
        ):
            with caplog.at_level(logging.WARNING, logger="backpropagate.export"):
                shelf = list_adapter_shelf("llama3.2")
        assert shelf == []
        assert any("ollama serve" in r.message for r in caplog.records)

    def test_empty_on_timeout(self):
        from backpropagate.export import list_adapter_shelf

        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ollama", 30),
        ):
            shelf = list_adapter_shelf("llama3.2")
        assert shelf == []

    def test_no_match_returns_empty(self):
        from backpropagate.export import list_adapter_shelf

        mock_result = MagicMock()
        mock_result.stdout = (
            "NAME              ID        SIZE      MODIFIED\n"
            "qwen2.5:taskA     abc       4.7 GB    2 days ago\n"
        )
        mock_result.returncode = 0
        with patch("shutil.which", return_value="/usr/bin/ollama"), patch(
            "subprocess.run", return_value=mock_result
        ):
            shelf = list_adapter_shelf("llama3.2")
        assert shelf == []


class TestOllamaAdapterFormatEnum:
    """The new ExportFormat value exists alongside the existing ones."""

    def test_ollama_adapter_format_value(self):
        from backpropagate.export import ExportFormat

        assert ExportFormat.OLLAMA_ADAPTER.value == "ollama-adapter"
        # The existing values are untouched.
        assert ExportFormat.LORA.value == "lora"
        assert ExportFormat.MERGED.value == "merged"
        assert ExportFormat.GGUF.value == "gguf"

    def test_export_result_summary_includes_adapter_format(self, temp_dir):
        from backpropagate.export import ExportFormat, ExportResult

        result = ExportResult(
            format=ExportFormat.OLLAMA_ADAPTER,
            path=temp_dir / "Modelfile",
            size_mb=0.001,
            quantization="llama3.2:taskA",
        )
        summary = result.summary()
        assert "ollama-adapter" in summary.lower()
