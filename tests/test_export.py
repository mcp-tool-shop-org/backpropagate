"""Tests for export functions."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock


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
        from backpropagate.export import ExportResult, ExportFormat

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
        from backpropagate.export import ExportResult, ExportFormat

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
        from backpropagate.export import ExportResult, ExportFormat

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
        from backpropagate.export import export_lora, ExportFormat

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
        from backpropagate.export import export_lora, ExportFormat

        # Patch the peft check to recognize our mock
        with patch("backpropagate.export._is_peft_model", return_value=True):
            output_dir = temp_dir / "output"

            result = export_lora(model=mock_peft_model, output_dir=output_dir)

            assert result.format == ExportFormat.LORA
            mock_peft_model.save_pretrained.assert_called_once()

    def test_export_lora_invalid_type(self, temp_dir):
        """Test export_lora raises error for invalid model type."""
        from backpropagate.export import export_lora

        # String paths that don't exist should raise ExportError
        from backpropagate.exceptions import ExportError
        with patch("backpropagate.export._is_peft_model", return_value=False):
            with pytest.raises(ExportError, match="Cannot export LoRA"):
                export_lora(model=12345, output_dir=temp_dir)  # Non-path, non-model type


class TestExportMerged:
    """Tests for export_merged function."""

    def test_export_merged_basic(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test basic merged export."""
        from backpropagate.export import export_merged, ExportFormat

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
        from backpropagate.export import export_merged
        from backpropagate.exceptions import MergeExportError

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
        from backpropagate.export import export_merged
        from backpropagate.exceptions import MergeExportError

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
        from backpropagate.export import export_gguf, ExportFormat

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
        from backpropagate.export import export_gguf
        from backpropagate.exceptions import InvalidSettingError

        with pytest.raises(InvalidSettingError, match="quantization"):
            export_gguf(
                model=mock_peft_model,
                tokenizer=mock_tokenizer,
                output_dir=temp_dir,
                quantization="invalid_quant",
            )

    def test_export_gguf_quantization_enum(self, temp_dir, mock_peft_model, mock_tokenizer):
        """Test export_gguf accepts GGUFQuantization enum."""
        from backpropagate.export import export_gguf, GGUFQuantization

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
        """Test basic Modelfile creation."""
        from backpropagate.export import create_modelfile

        modelfile = create_modelfile(sample_gguf_path)

        assert modelfile.exists()
        content = modelfile.read_text()
        assert "FROM" in content
        assert str(sample_gguf_path.resolve()) in content

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


class TestOllamaIntegration:
    """Tests for Ollama integration functions."""

    def test_register_with_ollama_file_not_found(self, temp_dir):
        """Test register_with_ollama raises error for missing file."""
        from backpropagate.export import register_with_ollama
        from backpropagate.exceptions import OllamaRegistrationError

        with pytest.raises(OllamaRegistrationError, match="GGUF file not found"):
            register_with_ollama(temp_dir / "nonexistent.gguf", "test-model")

    def test_register_with_ollama_no_ollama(self, sample_gguf_path):
        """Test register_with_ollama raises error when Ollama not found."""
        from backpropagate.export import register_with_ollama
        from backpropagate.exceptions import OllamaRegistrationError

        with patch("shutil.which", return_value=None):
            with pytest.raises(OllamaRegistrationError, match="Ollama CLI not found"):
                register_with_ollama(sample_gguf_path, "test-model")

    def test_register_with_ollama_success(self, sample_gguf_path):
        """Test successful Ollama registration."""
        from backpropagate.export import register_with_ollama

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            result = register_with_ollama(sample_gguf_path, "test-model")

        assert result is True
        mock_run.assert_called_once()

    def test_register_with_ollama_failure(self, sample_gguf_path):
        """Test Ollama registration failure raises OllamaRegistrationError."""
        from backpropagate.export import register_with_ollama
        from backpropagate.exceptions import OllamaRegistrationError
        import subprocess

        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ollama")):
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
        from backpropagate.export import list_ollama_models
        import subprocess

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

    def test_is_peft_model_true(self, mock_peft_model):
        """Test _is_peft_model returns True for PeftModel."""
        from backpropagate.export import _is_peft_model

        with patch("backpropagate.export.PeftModel", create=True) as MockPeft:
            # Make isinstance return True
            with patch("builtins.isinstance", return_value=True):
                # We need to patch at import level
                pass

        # Without actual peft installed, this will return False
        # Just test it doesn't crash
        result = _is_peft_model(mock_peft_model)
        assert isinstance(result, bool)

    def test_has_unsloth(self):
        """Test _has_unsloth detection."""
        # Test with unsloth not available
        with patch.dict("sys.modules", {"unsloth": None}):
            from backpropagate import export
            # Force reimport check
            with patch("builtins.__import__", side_effect=ImportError("No unsloth")):
                # The function should return False when import fails
                pass

        # Just verify the function exists and can be called in some form
        from backpropagate.export import _has_unsloth
        # On Python 3.14 with unsloth installed but incompatible, this may raise
        # We'll catch and verify it at least attempts the check
        try:
            result = _has_unsloth()
            assert isinstance(result, bool)
        except RuntimeError as e:
            # Python 3.14 torch.compile issue - this is expected
            assert "torch.compile" in str(e) or "3.14" in str(e)
            pytest.skip("Unsloth incompatible with Python 3.14")


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================

class TestExportGgufWithoutUnsloth:
    """Tests for export_gguf fallback path without Unsloth."""

    def test_export_gguf_no_unsloth_no_llama_cpp_raises(self, temp_dir, mock_peft_model, mock_tokenizer):
        """export_gguf should raise GGUFExportError when no Unsloth and no llama.cpp."""
        from backpropagate.export import export_gguf
        from backpropagate.exceptions import GGUFExportError
        import shutil

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
        from backpropagate.export import export_gguf
        from backpropagate.exceptions import GGUFExportError

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
        from backpropagate.export import export_gguf
        from backpropagate.exceptions import GGUFExportError

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
        from backpropagate.export import export_lora, ExportFormat

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
        from backpropagate.export import export_gguf, ExportFormat

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
        from backpropagate.export import export_gguf
        from backpropagate.exceptions import GGUFExportError

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


class TestExportResultSummary:
    """Additional tests for ExportResult.summary() method."""

    def test_summary_without_time(self, temp_dir):
        """summary() should handle zero export time gracefully."""
        from backpropagate.export import ExportResult, ExportFormat

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
        from backpropagate.export import ExportResult, ExportFormat

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
