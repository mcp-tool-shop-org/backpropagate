"""
Backpropagate - Model Export
============================

Export fine-tuned models to various formats.

Formats:
- LoRA adapter (default)
- Merged model (base + adapter)
- GGUF (for llama.cpp, Ollama, LM Studio)

GGUF Quantizations (fastest → smallest):
- f16: Full precision (largest)
- q8_0: 8-bit (best quality)
- q4_k_m: 4-bit (recommended balance)
- q4_0: 4-bit (fastest, lower quality)
"""

import logging
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from .exceptions import (
    ExportError,
    GGUFExportError,
    MergeExportError,
    OllamaRegistrationError,
    InvalidSettingError,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


__all__ = [
    "GGUFQuantization",
    "ExportFormat",
    "ExportResult",
    "export_lora",
    "export_merged",
    "export_gguf",
    "create_modelfile",
    "register_with_ollama",
    "list_ollama_models",
]


class GGUFQuantization(Enum):
    """GGUF quantization levels (fastest → smallest)."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q5_K_M = "q5_k_m"
    Q4_K_M = "q4_k_m"
    Q4_0 = "q4_0"
    Q2_K = "q2_k"


class ExportFormat(Enum):
    """Model export formats."""

    LORA = "lora"
    MERGED = "merged"
    GGUF = "gguf"


@dataclass
class ExportResult:
    """Result of a model export operation."""

    format: ExportFormat
    path: Path
    size_mb: float
    quantization: Optional[str] = None  # For GGUF
    export_time_seconds: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of the export."""
        lines = [
            f"Export Complete",
            f"  Format: {self.format.value}",
            f"  Path: {self.path}",
            f"  Size: {self.size_mb:.1f} MB",
        ]
        if self.quantization:
            lines.append(f"  Quantization: {self.quantization}")
        if self.export_time_seconds > 0:
            lines.append(f"  Time: {self.export_time_seconds:.1f}s")
        return "\n".join(lines)


def _get_dir_size_mb(path: Path) -> float:
    """Get total size of directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def _is_peft_model(model: Any) -> bool:
    """Check if model is a PeftModel."""
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False


def _has_unsloth() -> bool:
    """Check if Unsloth is available."""
    try:
        # Suppress import order warning - expected when checking availability
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
            import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def export_lora(
    model: Any,
    output_dir: Union[str, Path],
    adapter_name: str = "default",
) -> ExportResult:
    """
    Export LoRA adapter only.

    Args:
        model: PeftModel or path to saved model
        output_dir: Directory to save adapter
        adapter_name: Name of adapter to save

    Returns:
        ExportResult with path and size info

    Raises:
        ExportError: If export fails
    """
    start_time = time.time()
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise ExportError(
            f"Cannot create output directory: {e}",
            suggestion=f"Check write permissions for {output_path.parent}"
        ) from e
    except OSError as e:
        raise ExportError(f"Failed to create output directory: {e}") from e

    try:
        if isinstance(model, (str, Path)):
            # Copy existing adapter
            src_path = Path(model)
            if not src_path.exists():
                raise ExportError(
                    f"Source model path does not exist: {src_path}",
                    suggestion="Check that the model was trained and saved correctly"
                )
            if src_path.is_dir():
                # Copy adapter files
                files_copied = 0
                for pattern in ["adapter_*.safetensors", "adapter_*.bin", "adapter_config.json"]:
                    for f in src_path.glob(pattern):
                        shutil.copy2(f, output_path / f.name)
                        files_copied += 1
                if files_copied == 0:
                    raise ExportError(
                        f"No adapter files found in {src_path}",
                        suggestion="Ensure the directory contains adapter_*.safetensors or adapter_*.bin files"
                    )
        elif _is_peft_model(model):
            # Save from PeftModel
            model.save_pretrained(output_path, adapter_name=adapter_name)
        else:
            raise ExportError(
                f"Cannot export LoRA from {type(model).__name__}",
                suggestion="Expected PeftModel or path to saved adapter"
            )
    except ExportError:
        raise
    except Exception as e:
        raise ExportError(f"LoRA export failed: {e}") from e

    export_time = time.time() - start_time
    size_mb = _get_dir_size_mb(output_path)

    logger.info(f"LoRA adapter exported to {output_path} ({size_mb:.1f} MB)")

    return ExportResult(
        format=ExportFormat.LORA,
        path=output_path,
        size_mb=size_mb,
        export_time_seconds=export_time,
    )


def export_merged(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    output_dir: Union[str, Path],
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
) -> ExportResult:
    """
    Merge adapter into base model and save.

    Args:
        model: PeftModel with adapter
        tokenizer: Tokenizer to save with model
        output_dir: Directory to save merged model
        push_to_hub: Whether to push to Hugging Face Hub
        repo_id: Repository ID for Hub (required if push_to_hub=True)

    Returns:
        ExportResult with path and size info

    Raises:
        MergeExportError: If merge or save fails
    """
    start_time = time.time()
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise MergeExportError(
            f"Cannot create output directory: {e}",
            suggestion=f"Check write permissions for {output_path.parent}"
        ) from e
    except OSError as e:
        raise MergeExportError(f"Failed to create output directory: {e}") from e

    if not _is_peft_model(model):
        raise MergeExportError(
            f"Cannot merge non-PeftModel (got {type(model).__name__})",
            suggestion="Ensure you're exporting a model with LoRA adapters applied"
        )

    try:
        # Merge and unload adapter
        merged_model = model.merge_and_unload()
    except Exception as e:
        raise MergeExportError(f"Failed to merge LoRA adapters: {e}") from e

    try:
        # Save model and tokenizer
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        raise MergeExportError(f"Failed to save merged model: {e}") from e

    # Push to Hub if requested
    if push_to_hub:
        if not repo_id:
            raise MergeExportError(
                "repo_id required when push_to_hub=True",
                suggestion="Provide repo_id='username/model-name'"
            )
        try:
            merged_model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            logger.info(f"Pushed merged model to HuggingFace Hub: {repo_id}")
        except Exception as e:
            raise MergeExportError(
                f"Failed to push to HuggingFace Hub: {e}",
                suggestion="Check your HuggingFace token and repo_id"
            ) from e

    export_time = time.time() - start_time
    size_mb = _get_dir_size_mb(output_path)

    logger.info(f"Merged model exported to {output_path} ({size_mb:.1f} MB)")

    return ExportResult(
        format=ExportFormat.MERGED,
        path=output_path,
        size_mb=size_mb,
        export_time_seconds=export_time,
    )


def export_gguf(
    model: Any,
    tokenizer: "PreTrainedTokenizer",
    output_dir: Union[str, Path],
    quantization: Union[str, GGUFQuantization] = "q4_k_m",
    model_name: Optional[str] = None,
) -> ExportResult:
    """
    Export to GGUF format.

    Uses Unsloth's save_pretrained_gguf if available (much faster).
    Falls back to manual conversion via llama.cpp if needed.

    Args:
        model: Model to export (PeftModel or base model)
        tokenizer: Tokenizer for the model
        output_dir: Directory to save GGUF file
        quantization: Quantization level (default: q4_k_m)
        model_name: Name for the output file (default: "model")

    Returns:
        ExportResult with path and size info

    Raises:
        GGUFExportError: If GGUF export fails
        InvalidSettingError: If quantization is invalid
    """
    start_time = time.time()
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise GGUFExportError(
            f"Cannot create output directory: {e}",
            output_path=str(output_path),
        ) from e
    except OSError as e:
        raise GGUFExportError(
            f"Failed to create output directory: {e}",
            output_path=str(output_path),
        ) from e

    # Normalize quantization
    if isinstance(quantization, GGUFQuantization):
        quant_str = quantization.value
    else:
        quant_str = quantization.lower()

    # Validate quantization
    valid_quants = {q.value for q in GGUFQuantization}
    if quant_str not in valid_quants:
        raise InvalidSettingError(
            "quantization",
            quant_str,
            f"one of {sorted(valid_quants)}",
            suggestion="Try 'q4_k_m' for a good balance of size and quality"
        )

    model_name = model_name or "model"

    # Try Unsloth first (fastest)
    if _has_unsloth():
        try:
            # Unsloth handles everything
            model.save_pretrained_gguf(
                str(output_path),
                tokenizer,
                quantization_method=quant_str,
            )
            # Find the generated GGUF file
            gguf_files = list(output_path.glob("*.gguf"))
            if gguf_files:
                gguf_path = gguf_files[0]
            else:
                gguf_path = output_path / f"{model_name}-{quant_str}.gguf"

            # Validate output exists
            if not gguf_path.exists():
                raise GGUFExportError(
                    f"GGUF file was not created at expected path: {gguf_path}",
                    output_path=str(output_path),
                    quantization=quant_str,
                    suggestion="Check Unsloth logs for conversion errors"
                )

            export_time = time.time() - start_time
            size_mb = _get_dir_size_mb(gguf_path)

            if size_mb == 0:
                raise GGUFExportError(
                    f"GGUF file is empty (0 bytes): {gguf_path}",
                    output_path=str(output_path),
                    quantization=quant_str,
                )

            logger.info(f"GGUF exported via Unsloth to {gguf_path} ({size_mb:.1f} MB)")

            return ExportResult(
                format=ExportFormat.GGUF,
                path=gguf_path,
                size_mb=size_mb,
                quantization=quant_str,
                export_time_seconds=export_time,
            )
        except GGUFExportError:
            raise
        except Exception as e:
            # Fall through to manual conversion
            logger.warning(f"Unsloth GGUF export failed: {e}. Trying manual conversion...")

    # Manual conversion: merge first, then convert
    # This requires llama.cpp's convert script
    merged_path = output_path / "merged_temp"

    try:
        merged_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise GGUFExportError(
            f"Failed to create temp directory for merge: {e}",
            output_path=str(output_path),
        ) from e

    try:
        # Merge if needed
        if _is_peft_model(model):
            merged_model = model.merge_and_unload()
        else:
            merged_model = model

        # Save in HF format
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(merged_path, ignore_errors=True)
        raise GGUFExportError(
            f"Failed to prepare model for GGUF conversion: {e}",
            output_path=str(output_path),
        ) from e

    # Try to find llama.cpp convert script
    gguf_path = output_path / f"{model_name}-{quant_str}.gguf"

    # Check for llama-cpp-python or llama.cpp
    convert_script = None
    llama_cpp_paths = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("llama.cpp") / "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
    ]

    for path in llama_cpp_paths:
        if path.exists():
            convert_script = path
            break

    if convert_script:
        # Run conversion
        cmd = [
            "python",
            str(convert_script),
            str(merged_path),
            "--outfile",
            str(gguf_path),
            "--outtype",
            quant_str,
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Clean up on failure
            shutil.rmtree(merged_path, ignore_errors=True)
            error_output = e.stderr[:500] if e.stderr else "No error output"
            raise GGUFExportError(
                f"llama.cpp conversion failed (exit code {e.returncode}):\n{error_output}",
                output_path=str(output_path),
                quantization=quant_str,
                suggestion="Check that llama.cpp is properly installed and up to date"
            ) from e
    else:
        # Clean up and raise
        shutil.rmtree(merged_path, ignore_errors=True)
        raise GGUFExportError(
            "GGUF export requires either Unsloth or llama.cpp",
            output_path=str(output_path),
            suggestion=(
                "Install Unsloth: pip install unsloth\n"
                "Or clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp"
            )
        )

    # Clean up temp merged model
    try:
        shutil.rmtree(merged_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temp directory {merged_path}: {e}")

    # Validate output
    if not gguf_path.exists():
        raise GGUFExportError(
            f"GGUF file was not created: {gguf_path}",
            output_path=str(output_path),
            quantization=quant_str,
        )

    export_time = time.time() - start_time
    size_mb = _get_dir_size_mb(gguf_path)

    logger.info(f"GGUF exported via llama.cpp to {gguf_path} ({size_mb:.1f} MB)")

    return ExportResult(
        format=ExportFormat.GGUF,
        path=gguf_path,
        size_mb=size_mb,
        quantization=quant_str,
        export_time_seconds=export_time,
    )


def create_modelfile(
    gguf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    context_length: int = 4096,
) -> Path:
    """
    Create Ollama Modelfile for the GGUF.

    Args:
        gguf_path: Path to the GGUF file
        output_path: Path to write Modelfile (default: same dir as GGUF)
        system_prompt: Optional system prompt
        temperature: Model temperature (default: 0.7)
        context_length: Context window size (default: 4096)

    Returns:
        Path to the created Modelfile
    """
    gguf_path = Path(gguf_path).resolve()

    if output_path:
        modelfile_path = Path(output_path)
    else:
        modelfile_path = gguf_path.parent / "Modelfile"

    lines = [
        f'FROM "{gguf_path}"',
        "",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER num_ctx {context_length}",
    ]

    if system_prompt:
        # Escape quotes in system prompt
        escaped_prompt = system_prompt.replace('"', '\\"')
        lines.extend(
            [
                "",
                f'SYSTEM "{escaped_prompt}"',
            ]
        )

    modelfile_path.write_text("\n".join(lines))
    return modelfile_path


def register_with_ollama(
    gguf_path: Union[str, Path],
    model_name: str,
    system_prompt: Optional[str] = None,
) -> bool:
    """
    Register GGUF with Ollama.

    Creates Modelfile and runs `ollama create`.

    Args:
        gguf_path: Path to the GGUF file
        model_name: Name for the Ollama model
        system_prompt: Optional system prompt

    Returns:
        True if successful

    Raises:
        OllamaRegistrationError: If registration fails
    """
    gguf_path = Path(gguf_path).resolve()

    if not gguf_path.exists():
        raise OllamaRegistrationError(
            model_name,
            f"GGUF file not found: {gguf_path}",
            suggestion="Check that the GGUF export completed successfully"
        )

    # Check if Ollama is available
    if not shutil.which("ollama"):
        raise OllamaRegistrationError(
            model_name,
            "Ollama CLI not found in PATH",
            suggestion="Install Ollama from https://ollama.ai and ensure it's in your PATH"
        )

    # Create Modelfile
    try:
        modelfile_path = create_modelfile(gguf_path, system_prompt=system_prompt)
    except Exception as e:
        raise OllamaRegistrationError(
            model_name,
            f"Failed to create Modelfile: {e}",
        ) from e

    try:
        # Run ollama create
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Successfully registered model '{model_name}' with Ollama")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[:500] if e.stderr else "Unknown error"
        raise OllamaRegistrationError(
            model_name,
            f"ollama create failed: {error_msg}",
            suggestion="Ensure Ollama is running (ollama serve) and try again"
        ) from e
    finally:
        # Clean up Modelfile
        try:
            if modelfile_path.exists():
                modelfile_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up Modelfile: {e}")


def list_ollama_models() -> List[str]:
    """
    List models registered with Ollama.

    Returns:
        List of model names
    """
    if not shutil.which("ollama"):
        return []

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse output (skip header line)
        models = []
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:  # Skip header
            if line.strip():
                # First column is model name
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    except subprocess.CalledProcessError:
        return []
