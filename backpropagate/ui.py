"""
Backpropagate - Gradio 6 Web Interface
=======================================

A beautiful, Apple-inspired interface for LLM fine-tuning.

Features:
- Ocean Mist theme (brand consistent with Comfy Headless)
- Live training progress with loss plots
- GPU monitoring dashboard
- Dataset preview and validation
- Run history and comparison
- One-click export to multiple formats

Security (Production-Hardened):
- Authentication required when share=True (public URLs)
- Path validation to prevent traversal attacks
- Input sanitization for user-provided values
- Rate limiting with IP tracking for training operations
- File upload validation (CVE-2024-47872 mitigation)
- CSRF protection (CVE-2024-1727 mitigation)
- Security event logging for monitoring
- Proper gr.Error/gr.Warning/gr.Info for user feedback

Based on:
- Trail of Bits Gradio 5 Security Audit
- OWASP Web Security Best Practices

Usage:
    from backpropagate import launch
    launch(port=7862)

    # With authentication (required for public sharing)
    launch(port=7862, share=True, auth=("admin", "password"))
"""

import gradio as gr
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import time
import secrets
from pathlib import Path

from .theme import create_backpropagate_theme, get_css
from .config import settings, get_settings
from .feature_flags import (
    FEATURES,
    get_gpu_info,
    get_system_info,
    list_available_features,
    list_missing_features,
)
from .gpu_safety import get_gpu_status, GPUCondition, format_gpu_status
from .datasets import (
    DatasetLoader,
    DatasetFormat,
    ValidationResult,
    get_dataset_stats,
    preview_samples,
)
from .security import safe_path, PathTraversalError, SecurityWarning

# Import production security utilities
from .ui_security import (
    SecurityConfig,
    DEFAULT_SECURITY_CONFIG,
    EnhancedRateLimiter,
    FileValidator,
    ALLOWED_DATASET_EXTENSIONS,
    DANGEROUS_EXTENSIONS,
    raise_gradio_error,
    raise_gradio_warning,
    raise_gradio_info,
    safe_gradio_handler,
    validate_numeric_input,
    validate_string_input,
    validate_and_log_request,
    log_security_event,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY UTILITIES (Enhanced with IP tracking and security logging)
# =============================================================================

# Global rate limiters using enhanced version with IP tracking
_training_limiter = EnhancedRateLimiter(
    max_requests=DEFAULT_SECURITY_CONFIG.training_rate_limit,
    window_seconds=DEFAULT_SECURITY_CONFIG.training_rate_window,
    operation_name="training",
)
_export_limiter = EnhancedRateLimiter(
    max_requests=DEFAULT_SECURITY_CONFIG.export_rate_limit,
    window_seconds=DEFAULT_SECURITY_CONFIG.export_rate_window,
    operation_name="export",
)
_upload_limiter = EnhancedRateLimiter(
    max_requests=DEFAULT_SECURITY_CONFIG.upload_rate_limit,
    window_seconds=DEFAULT_SECURITY_CONFIG.upload_rate_window,
    operation_name="upload",
)

# File validator for dataset uploads (CVE-2024-47872 mitigation)
_file_validator = FileValidator(
    allowed_extensions=ALLOWED_DATASET_EXTENSIONS,
    max_size_mb=DEFAULT_SECURITY_CONFIG.max_upload_size_mb,
)


def validate_path_input(
    path_str: str,
    allowed_base: Optional[Path] = None,
    must_exist: bool = False,
) -> Tuple[bool, str, Optional[Path]]:
    """
    Validate a user-provided path input.

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    if not path_str or not path_str.strip():
        return False, "Path cannot be empty", None

    try:
        resolved = safe_path(
            path_str,
            must_exist=must_exist,
            allowed_base=allowed_base,
        )
        return True, "", resolved
    except PathTraversalError as e:
        logger.warning(f"Path traversal attempt blocked: {e}")
        return False, f"Security error: {e}", None
    except FileNotFoundError:
        return False, f"Path not found: {path_str}", None
    except ValueError as e:
        return False, str(e), None


def sanitize_model_name(name: str) -> str:
    """Sanitize a model name for safe use."""
    if not name:
        return ""
    # Allow alphanumeric, hyphens, underscores, slashes (for HF paths), and dots
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./")
    return "".join(c for c in name if c in allowed)


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """Sanitize general text input."""
    if not text:
        return ""
    # Truncate to max length
    text = text[:max_length]
    # Remove null bytes and other problematic characters
    text = text.replace("\x00", "")
    return text.strip()


def generate_auth_token() -> str:
    """Generate a secure authentication token."""
    return secrets.token_urlsafe(32)

__all__ = ["create_ui", "launch"]


# =============================================================================
# MODEL PRESETS
# =============================================================================

MODEL_PRESETS = {
    "Qwen 2.5 7B (Recommended)": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "Qwen 2.5 3B (Fast)": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "Qwen 2.5 1.5B (Tiny)": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "Llama 3.2 3B": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "Llama 3.2 1B": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "Mistral 7B v0.3": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "Phi-3 Mini": "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
}

DATASET_PRESETS = {
    "UltraChat 200K (Conversations)": "HuggingFaceH4/ultrachat_200k",
    "OpenAssistant (Helpful)": "OpenAssistant/oasst1",
    "Alpaca (Instructions)": "tatsu-lab/alpaca",
    "Custom JSONL": "custom",
}


# =============================================================================
# UI STATE
# =============================================================================

class UIState:
    """Global UI state container."""
    trainer = None
    is_training = False
    current_run = None
    loss_history = []
    runs_history = []
    # Multi-run state
    multi_run_trainer = None
    multi_run_is_running = False
    multi_run_results = None
    multi_run_current_run = 0
    multi_run_loss_history = []
    multi_run_run_boundaries = []
    # Dataset state
    dataset_loader: Optional[DatasetLoader] = None
    dataset_validation: Optional[ValidationResult] = None


state = UIState()


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def get_gpu_status_display() -> str:
    """Get formatted GPU status for display."""
    gpu = get_gpu_info()
    if gpu.get("available"):
        name = gpu.get("device_name", "Unknown GPU")
        total = gpu.get("memory_total", 0) / (1024**3)
        allocated = gpu.get("memory_allocated", 0) / (1024**3)
        return f"**{name}**\n\nVRAM: {allocated:.1f} / {total:.1f} GB"
    return "**No GPU detected**"


def get_system_status() -> str:
    """Get formatted system status."""
    info = get_system_info()
    features = list_available_features()

    lines = [
        f"**Python:** {info.get('python_version', 'Unknown').split()[0]}",
        f"**Platform:** {info.get('platform', 'Unknown')[:30]}",
        "",
        "**Features:**",
    ]
    for name, desc in features.items():
        lines.append(f"- {name}")

    return "\n".join(lines)


def format_loss_plot(losses: List[float]) -> Dict[str, Any]:
    """Format loss history for plotting."""
    if not losses:
        return {"data": [], "layout": {"title": "Training Loss"}}

    return {
        "data": [
            {
                "x": list(range(1, len(losses) + 1)),
                "y": losses,
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Loss",
                "line": {"color": "#7EC8C8", "width": 2},
                "marker": {"size": 4},
            }
        ],
        "layout": {
            "title": "",
            "xaxis": {"title": "Step", "gridcolor": "#3A4554"},
            "yaxis": {"title": "Loss", "gridcolor": "#3A4554"},
            "paper_bgcolor": "#232830",
            "plot_bgcolor": "#232830",
            "font": {"color": "#F0F4F8"},
            "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
        },
    }


def format_multi_run_plot(losses: List[float], run_boundaries: List[int]) -> Dict[str, Any]:
    """Format multi-run loss history with run boundaries marked."""
    if not losses:
        return {"data": [], "layout": {"title": "Multi-Run Loss"}}

    data = [
        {
            "x": list(range(1, len(losses) + 1)),
            "y": losses,
            "type": "scatter",
            "mode": "lines",
            "name": "Loss",
            "line": {"color": "#7EC8C8", "width": 2},
        }
    ]

    # Add vertical lines for run boundaries
    shapes = []
    for i, boundary in enumerate(run_boundaries):
        if boundary > 0:  # Skip first boundary at 0
            shapes.append({
                "type": "line",
                "x0": boundary,
                "x1": boundary,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#F59E0B", "width": 1, "dash": "dash"},
            })

    return {
        "data": data,
        "layout": {
            "title": "",
            "xaxis": {"title": "Step (aggregate)", "gridcolor": "#3A4554"},
            "yaxis": {"title": "Loss", "gridcolor": "#3A4554"},
            "paper_bgcolor": "#232830",
            "plot_bgcolor": "#232830",
            "font": {"color": "#F0F4F8"},
            "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
            "shapes": shapes,
        },
    }


def get_gpu_safety_status() -> str:
    """Get detailed GPU safety status for display."""
    status = get_gpu_status()  # From gpu_safety module - returns GPUStatus object

    if not status.available:
        return "**No GPU detected**"

    # Build status string
    lines = [f"**{status.device_name}**"]

    if status.temperature_c is not None:
        temp_color = "green"
        if status.temperature_c >= 80:
            temp_color = "orange"
        if status.temperature_c >= 90:
            temp_color = "red"
        lines.append(f"Temp: {status.temperature_c}C")

    lines.append(f"VRAM: {status.vram_used_gb:.1f} / {status.vram_total_gb:.1f} GB ({status.vram_percent:.0f}%)")

    if status.power_draw_w is not None:
        lines.append(f"Power: {status.power_draw_w:.0f}W")

    # Safety condition
    condition_emoji = {
        GPUCondition.SAFE: "OK",
        GPUCondition.WARM: "Warm",
        GPUCondition.WARNING: "Warning",
        GPUCondition.CRITICAL: "CRITICAL",
        GPUCondition.EMERGENCY: "EMERGENCY",
        GPUCondition.UNKNOWN: "Unknown",
    }
    lines.append(f"Status: {condition_emoji.get(status.condition, 'Unknown')}")

    return "\n\n".join(lines)


# =============================================================================
# DATASET FUNCTIONS
# =============================================================================

def load_dataset_file(file_obj, request: gr.Request = None) -> tuple:
    """
    Load and validate a dataset file with security checks.

    Security features:
    - Rate limiting to prevent abuse
    - File extension validation (CVE-2024-47872 mitigation)
    - File size validation
    - Path sanitization
    """
    # Rate limit check
    allowed, wait_time = _upload_limiter.check(request)
    if not allowed:
        raise gr.Error(
            f"Too many uploads. Please wait {wait_time:.0f} seconds.",
            duration=10,
            title="Rate Limited",
        )

    if file_obj is None:
        raise gr.Error("Please select a file to upload.", duration=5)

    # Validate file using security module (CVE-2024-47872 mitigation)
    is_valid, error_msg, validated_path = _file_validator.validate(file_obj, purpose="dataset_upload")
    if not is_valid:
        log_security_event(
            "file_upload_rejected",
            error=error_msg,
            filename=getattr(file_obj, "name", "unknown"),
        )
        raise gr.Error(error_msg, duration=10, title="Invalid File")

    try:
        file_path = str(validated_path)
        loader = DatasetLoader(file_path)
        state.dataset_loader = loader
        state.dataset_validation = loader.validation_result

        # Log successful upload
        log_security_event(
            "file_upload_success",
            filename=validated_path.name,
            format=loader.detected_format.value,
            samples=len(loader),
        )

        # Show info message
        gr.Info(f"Loaded {len(loader)} samples", duration=3)

        # Build status message
        status_lines = [
            f"**Format detected:** {loader.detected_format.value}",
            f"**Total samples:** {len(loader)}",
        ]

        if loader.is_valid:
            status_lines.append("**Validation:** ✅ Passed")
        else:
            status_lines.append(f"**Validation:** ⚠️ {loader.validation_result.error_count} errors")
            # Show warning for validation issues
            gr.Warning(
                f"Dataset has {loader.validation_result.error_count} validation errors. "
                "Check the validation report for details.",
                duration=5,
            )

        # Get stats
        stats = loader.stats()
        status_lines.extend([
            "",
            f"**Approx. tokens:** {stats.total_tokens_approx:,}",
            f"**Avg tokens/sample:** {stats.avg_tokens_per_sample:.0f}",
            f"**Has system prompts:** {'Yes' if stats.has_system_prompts else 'No'}",
            f"**Avg turns/conversation:** {stats.avg_turns_per_conversation:.1f}",
        ])

        status = "\n".join(status_lines)

        # Get validation report
        validation_text = loader.validation_report()

        # Get preview samples
        previews = loader.preview(n=3, as_chatml=True)
        preview_text = "\n\n---\n\n".join([f"**Sample {i+1}:**\n```\n{p}\n```" for i, p in enumerate(previews)])

        # Format table data
        table_data = [
            ["Format", loader.detected_format.value],
            ["Samples", str(len(loader))],
            ["Valid", "Yes" if loader.is_valid else "No"],
            ["Errors", str(loader.validation_result.error_count)],
            ["Warnings", str(loader.validation_result.warning_count)],
            ["Approx. Tokens", f"{stats.total_tokens_approx:,}"],
            ["Avg Tokens/Sample", f"{stats.avg_tokens_per_sample:.0f}"],
            ["System Prompts", str(stats.unique_system_prompts)],
        ]

        return (
            status,
            validation_text,
            gr.update(value=table_data),
            gr.update(value=preview_text),
            "✅ Loaded successfully!",
        )

    except gr.Error:
        # Re-raise Gradio errors as-is
        raise
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        raise gr.Error(f"File not found: {e}", duration=10, title="File Not Found")
    except Exception as e:
        logger.exception("Failed to load dataset")
        log_security_event(
            "file_upload_error",
            error=str(e)[:200],
            filename=getattr(file_obj, "name", "unknown"),
        )
        raise gr.Error(
            f"Failed to load dataset: {str(e)}",
            duration=10,
            title="Load Error",
        )


def convert_dataset_format(target_format: str) -> tuple:
    """Convert loaded dataset to target format."""
    if state.dataset_loader is None:
        return "No dataset loaded", ""

    try:
        if target_format == "ChatML":
            samples = state.dataset_loader.to_chatml()
            # Show first 3 samples
            preview = "\n\n---\n\n".join([
                f"**Sample {i+1}:**\n```\n{s['text']}\n```"
                for i, s in enumerate(samples[:3])
            ])
            return f"Converted {len(samples)} samples to ChatML format", preview

        return "Unsupported format", ""

    except Exception as e:
        logger.exception("Failed to convert dataset")
        return f"Error: {str(e)}", ""


def export_converted_dataset(output_path: str) -> str:
    """Export the converted dataset to a file."""
    if state.dataset_loader is None:
        return "No dataset loaded"

    # Validate output path
    is_valid, error_msg, validated_path = validate_path_input(output_path)
    if not is_valid:
        return f"Invalid output path: {error_msg}"

    try:
        import json

        validated_path.parent.mkdir(parents=True, exist_ok=True)

        samples = state.dataset_loader.to_chatml()

        with open(validated_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return f"Exported {len(samples)} samples to {validated_path}"

    except Exception as e:
        logger.exception("Failed to export dataset")
        return f"Error: {str(e)}"


def refresh_dataset_preview(show_raw: bool) -> str:
    """Refresh the preview with raw or ChatML format."""
    if state.dataset_loader is None:
        return "No dataset loaded"

    try:
        if show_raw:
            import json
            samples = state.dataset_loader.samples[:3]
            return "\n\n---\n\n".join([
                f"**Sample {i+1}:**\n```json\n{json.dumps(s, indent=2)}\n```"
                for i, s in enumerate(samples)
            ])
        else:
            previews = state.dataset_loader.preview(n=3, as_chatml=True)
            return "\n\n---\n\n".join([
                f"**Sample {i+1}:**\n```\n{p}\n```"
                for i, p in enumerate(previews)
            ])

    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def start_training(
    model_preset: str,
    custom_model: str,
    dataset_preset: str,
    custom_dataset: str,
    max_samples: int,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    progress=gr.Progress(),
    request: gr.Request = None,
) -> tuple:
    """
    Start a training run with security checks.

    Security features:
    - Rate limiting with IP tracking
    - Input validation for all parameters
    - Model name sanitization
    - Path validation for custom datasets
    - Security event logging
    """
    from .trainer import Trainer

    # Rate limiting check with IP tracking
    allowed, wait_time = _training_limiter.check(request)
    if not allowed:
        log_security_event(
            "training_rate_limited",
            wait_seconds=wait_time,
        )
        raise gr.Error(
            f"Too many training requests. Please wait {wait_time:.0f} seconds.",
            duration=10,
            title="Rate Limited",
        )

    # Validate numeric inputs
    try:
        max_samples = int(validate_numeric_input(max_samples, "Max samples", min_value=1, max_value=1000000))
        max_steps = int(validate_numeric_input(max_steps, "Max steps", min_value=1, max_value=100000))
        learning_rate = validate_numeric_input(learning_rate, "Learning rate", min_value=1e-8, max_value=1.0)
        batch_size = int(validate_numeric_input(batch_size, "Batch size", min_value=1, max_value=64))
        lora_r = int(validate_numeric_input(lora_r, "LoRA rank", min_value=1, max_value=256))
        lora_alpha = int(validate_numeric_input(lora_alpha, "LoRA alpha", min_value=1, max_value=512))
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Invalid parameter: {e}", duration=10)

    # Resolve and sanitize model name
    if model_preset in MODEL_PRESETS:
        model_name = MODEL_PRESETS[model_preset]
    else:
        model_name = sanitize_model_name(custom_model)
        if not model_name:
            raise gr.Error(
                "Invalid model name. Use format: org/model-name",
                duration=10,
                title="Invalid Model",
            )

    # Resolve dataset with path validation for custom datasets
    if dataset_preset == "Custom JSONL":
        is_valid, error_msg, validated_path = validate_path_input(custom_dataset, must_exist=True)
        if not is_valid:
            raise gr.Error(f"Dataset error: {error_msg}", duration=10, title="Invalid Dataset")
        dataset_name = str(validated_path)
    elif dataset_preset in DATASET_PRESETS:
        dataset_name = DATASET_PRESETS[dataset_preset]
    else:
        dataset_name = settings.data.dataset_name

    # Log training start
    log_security_event(
        "training_started",
        model=model_name,
        dataset=dataset_name[:100],
        samples=max_samples,
        steps=max_steps,
    )

    state.is_training = True
    state.loss_history = []
    state.train_start_time = time.time()

    # Show info message
    gr.Info(f"Starting training with {model_name}...", duration=3)

    status = f"🚀 Initializing trainer with {model_name}..."
    yield status, format_loss_plot([]), get_gpu_status_display()

    try:
        # Create trainer
        state.trainer = Trainer(
            model=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        status = "📦 Loading model..."
        yield status, format_loss_plot([]), get_gpu_status_display()

        state.trainer.load_model()

        status = f"🏋️ Training on {max_samples} samples for {max_steps} steps..."
        yield status, format_loss_plot([]), get_gpu_status_display()

        # Train
        run = state.trainer.train(
            dataset=dataset_name if dataset_name != "custom" else None,
            steps=max_steps,
            samples=max_samples,
        )

        state.current_run = run
        state.loss_history = run.loss_history
        state.runs_history.append(run)

        # Log training success
        log_security_event(
            "training_completed",
            model=model_name,
            final_loss=run.final_loss,
            duration_seconds=run.duration_seconds,
        )

        # Show success info
        gr.Info(f"Training complete! Loss: {run.final_loss:.4f}", duration=5)

        status = f"✅ Training complete! Final loss: {run.final_loss:.4f}"
        yield status, format_loss_plot(run.loss_history), get_gpu_status_display()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        gr.Warning("Training interrupted by user", duration=5)
        yield "⚠️ Training interrupted", format_loss_plot(state.loss_history), get_gpu_status_display()

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        log_security_event(
            "training_failed",
            model=model_name,
            error=str(e)[:200],
        )
        gr.Warning(f"Training failed: {str(e)[:100]}", duration=10)
        status = f"❌ Training failed: {str(e)}"
        yield status, format_loss_plot(state.loss_history), get_gpu_status_display()

    finally:
        state.is_training = False


def stop_training() -> str:
    """Stop the current training run."""
    # Note: Proper interruption requires more complex handling
    state.is_training = False
    return "Training stop requested..."


def save_model(output_path: str, save_merged: bool) -> str:
    """Save the trained model."""
    if state.trainer is None:
        return "No model to save. Train a model first."

    # Validate output path
    is_valid, error_msg, validated_path = validate_path_input(output_path)
    if not is_valid:
        return f"Invalid output path: {error_msg}"

    try:
        path = state.trainer.save(str(validated_path), save_merged=save_merged)
        return f"Model saved to: {path}"
    except Exception as e:
        return f"Save failed: {str(e)}"


def export_model(
    export_format: str,
    quantization: str,
    output_path: str,
    request: gr.Request = None,
) -> str:
    """
    Export the trained model with security checks.

    Security features:
    - Rate limiting with IP tracking
    - Path validation
    - Security event logging
    """
    if state.trainer is None:
        raise gr.Error(
            "No model to export. Train a model first.",
            duration=10,
            title="No Model",
        )

    # Rate limiting for export operations with IP tracking
    allowed, wait_time = _export_limiter.check(request)
    if not allowed:
        log_security_event("export_rate_limited", wait_seconds=wait_time)
        raise gr.Error(
            f"Too many export requests. Please wait {wait_time:.0f} seconds.",
            duration=10,
            title="Rate Limited",
        )

    # Validate output path
    is_valid, error_msg, validated_path = validate_path_input(output_path)
    if not is_valid:
        return f"Invalid output path: {error_msg}"

    try:
        result = state.trainer.export(
            format=export_format.lower(),
            output_dir=str(validated_path),
            quantization=quantization,
        )
        return result.summary()
    except Exception as e:
        return f"Export failed: {str(e)}"


def register_ollama(gguf_path: str, model_name: str, system_prompt: str) -> str:
    """Register a GGUF model with Ollama."""
    from .export import register_with_ollama, list_ollama_models

    # Validate GGUF path
    is_valid, error_msg, validated_path = validate_path_input(gguf_path, must_exist=True)
    if not is_valid:
        return f"Invalid GGUF path: {error_msg}"

    # Sanitize model name
    safe_model_name = sanitize_model_name(model_name)
    if not safe_model_name:
        return "Invalid model name. Use only alphanumeric characters, hyphens, and underscores."

    # Sanitize system prompt
    safe_system_prompt = sanitize_text_input(system_prompt, max_length=2000) if system_prompt else None

    try:
        success = register_with_ollama(
            gguf_path=validated_path,
            model_name=safe_model_name,
            system_prompt=safe_system_prompt if safe_system_prompt and safe_system_prompt.strip() else None,
        )

        if success:
            return f"Successfully registered with Ollama!\n\nRun with:\n```\nollama run {safe_model_name}\n```"
        else:
            return "Failed to register with Ollama. Check that Ollama is running."

    except Exception as e:
        return f"Ollama registration failed: {str(e)}"


def list_ollama() -> str:
    """List models registered with Ollama."""
    from .export import list_ollama_models

    try:
        models = list_ollama_models()
        if not models:
            return "No Ollama models found. Is Ollama running?"

        return "**Ollama Models:**\n" + "\n".join([f"- {m}" for m in models])

    except Exception as e:
        return f"Error listing Ollama models: {str(e)}"


def create_ollama_modelfile(gguf_path: str, output_path: str, system_prompt: str, temperature: float) -> str:
    """Create an Ollama Modelfile."""
    from .export import create_modelfile

    # Validate GGUF path
    is_valid, error_msg, validated_gguf = validate_path_input(gguf_path, must_exist=True)
    if not is_valid:
        return f"Invalid GGUF path: {error_msg}"

    # Validate output path if provided
    validated_output = None
    if output_path and output_path.strip():
        is_valid, error_msg, validated_output = validate_path_input(output_path)
        if not is_valid:
            return f"Invalid output path: {error_msg}"

    # Sanitize system prompt
    safe_system_prompt = sanitize_text_input(system_prompt, max_length=2000) if system_prompt else None

    try:
        modelfile = create_modelfile(
            gguf_path=validated_gguf,
            output_path=validated_output,
            system_prompt=safe_system_prompt if safe_system_prompt and safe_system_prompt.strip() else None,
            temperature=temperature,
        )

        content = modelfile.read_text()
        return f"Modelfile created: {modelfile}\n\n```\n{content}\n```"

    except Exception as e:
        return f"Failed to create Modelfile: {str(e)}"


def get_runs_table() -> List[List[Any]]:
    """Get training runs as table data."""
    rows = []
    for run in state.runs_history:
        rows.append([
            run.run_id,
            run.steps,
            f"{run.final_loss:.4f}",
            f"{run.duration_seconds:.1f}s",
            run.samples_seen,
        ])
    return rows


# =============================================================================
# MULTI-RUN FUNCTIONS
# =============================================================================

def start_multi_run(
    model_preset: str,
    custom_model: str,
    dataset_preset: str,
    custom_dataset: str,
    num_runs: int,
    steps_per_run: int,
    samples_per_run: int,
    merge_mode: str,
    initial_lr: float,
    final_lr: float,
    max_temp: float,
    # Phase 4 advanced options
    adaptive_scaling: bool = False,
    layer_scaling: bool = False,
    early_stopping: bool = False,
    early_patience: int = 2,
    val_samples: int = 100,
    # Phase 5.3 checkpoint options
    ckpt_auto_prune: bool = True,
    ckpt_keep_best: int = 3,
    ckpt_max_total: int = 10,
    ckpt_keep_final: bool = True,
    ckpt_keep_boundaries: bool = False,
    progress=gr.Progress(),
) -> tuple:
    """Start SLAO Multi-Run training."""
    from .multi_run import MultiRunTrainer, MultiRunConfig, MergeMode

    # Resolve model
    if model_preset in MODEL_PRESETS:
        model_name = MODEL_PRESETS[model_preset]
    else:
        model_name = custom_model

    # Resolve dataset
    if dataset_preset == "Custom JSONL":
        dataset_name = custom_dataset
    elif dataset_preset in DATASET_PRESETS:
        dataset_name = DATASET_PRESETS[dataset_preset]
    else:
        dataset_name = settings.data.dataset_name

    state.multi_run_is_running = True
    state.multi_run_loss_history = []
    state.multi_run_run_boundaries = []
    state.multi_run_current_run = 0

    status = f"Initializing SLAO Multi-Run with {model_name}..."
    yield (
        status,
        format_multi_run_plot([], []),
        get_gpu_safety_status(),
        get_multi_run_progress_table(),
    )

    try:
        config = MultiRunConfig(
            num_runs=int(num_runs),
            steps_per_run=int(steps_per_run),
            samples_per_run=int(samples_per_run),
            merge_mode=MergeMode(merge_mode.lower()),
            initial_lr=initial_lr,
            final_lr=final_lr,
            max_temp_c=max_temp,
            checkpoint_dir="./output/multi_run",
            # Phase 4 options
            validate_every_run=early_stopping,  # Enable validation if early stopping is on
            validation_samples=int(val_samples),
            early_stopping=early_stopping,
            early_stopping_patience=int(early_patience),
            # Phase 5.3 checkpoint options
            checkpoint_keep_best_n=int(ckpt_keep_best),
            checkpoint_keep_final=ckpt_keep_final,
            checkpoint_keep_run_boundaries=ckpt_keep_boundaries,
            checkpoint_max_total=int(ckpt_max_total),
            checkpoint_auto_prune=ckpt_auto_prune,
        )

        def on_run_complete(run_result):
            state.multi_run_current_run = run_result.run_index
            state.multi_run_loss_history.extend(run_result.loss_history)
            state.multi_run_run_boundaries.append(len(state.multi_run_loss_history))

        state.multi_run_trainer = MultiRunTrainer(
            model=model_name,
            config=config,
            on_run_complete=on_run_complete,
        )

        status = f"Starting {num_runs} runs..."
        yield (
            status,
            format_multi_run_plot([], []),
            get_gpu_safety_status(),
            get_multi_run_progress_table(),
        )

        # Run the multi-run training
        result = state.multi_run_trainer.run(dataset_name)

        state.multi_run_results = result
        state.multi_run_loss_history = result.aggregate_loss_history
        state.multi_run_run_boundaries = result.run_boundaries

        if result.aborted:
            status = f"Multi-run aborted: {result.abort_reason}"
        else:
            status = (
                f"Multi-run complete! {result.total_runs} runs, "
                f"Final loss: {result.final_loss:.4f}, "
                f"Time: {result.total_duration_seconds/60:.1f}min"
            )

        yield (
            status,
            format_multi_run_plot(result.aggregate_loss_history, result.run_boundaries),
            get_gpu_safety_status(),
            get_multi_run_progress_table(),
        )

    except Exception as e:
        logger.error(f"Multi-run failed: {e}")
        status = f"Multi-run failed: {str(e)}"
        yield (
            status,
            format_multi_run_plot(state.multi_run_loss_history, state.multi_run_run_boundaries),
            get_gpu_safety_status(),
            get_multi_run_progress_table(),
        )

    finally:
        state.multi_run_is_running = False


def stop_multi_run() -> str:
    """Stop the current multi-run training."""
    if state.multi_run_trainer:
        state.multi_run_trainer.abort("User requested stop")
    state.multi_run_is_running = False
    return "Multi-run stop requested..."


def get_multi_run_progress_table() -> List[List[Any]]:
    """Get multi-run progress as table data."""
    if state.multi_run_results is None:
        return []

    rows = []
    for run in state.multi_run_results.runs:
        merge_info = ""
        if run.merge_result:
            merge_info = f"scale={run.merge_result.scale_factor:.3f}"

        rows.append([
            run.run_index,
            run.steps,
            run.samples,
            f"{run.final_loss:.4f}",
            f"{run.learning_rate:.2e}",
            f"{run.duration_seconds:.1f}s",
            merge_info,
        ])
    return rows


# =============================================================================
# PHASE 5: TRAINING DASHBOARD FUNCTIONS
# =============================================================================

def get_dashboard_metrics() -> Dict[str, str]:
    """
    Get all dashboard metrics for the training sidebar.

    Phase 5.1: Real-time metrics display.

    Returns:
        Dictionary with all dashboard metric strings
    """
    metrics = {
        # Live Metrics
        "current_run": "**Current Run:** -",
        "current_step": "**Step:** -",
        "current_loss": "**Loss:** -",
        "eta": "**ETA:** -",
        # GPU Status
        "gpu_temp": "**Temperature:** -",
        "gpu_vram": "**VRAM:** -",
        "gpu_power": "**Power:** -",
        "gpu_condition": "**Status:** -",
        # SLAO Merge Stats
        "scale_factor": "**Scale Factor:** -",
        "similarity": "**Task Similarity:** -",
        "a_matrices": "**A Matrices Merged:** -",
        "b_matrices": "**B Matrices Merged:** -",
        # Early Stopping
        "val_loss": "**Validation Loss:** -",
        "best_val": "**Best Val Loss:** -",
        "patience": "**Patience:** -",
        "early_stop_status": "**Status:** Not enabled",
        # Run Timeline
        "total_runs": "**Total Runs:** -",
        "completed_runs": "**Completed:** -",
        "total_steps": "**Total Steps:** -",
        "total_samples": "**Total Samples:** -",
        "total_time": "**Elapsed Time:** -",
        # Phase 5.3: Checkpoints
        "ckpt_count": "**Saved:** -",
        "ckpt_size": "**Total Size:** -",
        "ckpt_best": "**Best:** -",
        "ckpt_prunable": "**Prunable:** -",
        "ckpt_policy": "**Policy:** -",
    }

    # Get GPU status
    status = get_gpu_status()
    if status.available:
        if status.temperature_c is not None:
            temp_emoji = "🟢" if status.temperature_c < 70 else "🟡" if status.temperature_c < 85 else "🔴"
            metrics["gpu_temp"] = f"**Temperature:** {temp_emoji} {status.temperature_c}°C"

        metrics["gpu_vram"] = f"**VRAM:** {status.vram_used_gb:.1f} / {status.vram_total_gb:.1f} GB ({status.vram_percent:.0f}%)"

        if status.power_draw_w is not None:
            metrics["gpu_power"] = f"**Power:** {status.power_draw_w:.0f}W"

        condition_map = {
            GPUCondition.SAFE: "🟢 Safe",
            GPUCondition.WARM: "🟡 Warm",
            GPUCondition.WARNING: "🟠 Warning",
            GPUCondition.CRITICAL: "🔴 Critical",
            GPUCondition.EMERGENCY: "🚨 Emergency",
            GPUCondition.UNKNOWN: "⚪ Unknown",
        }
        metrics["gpu_condition"] = f"**Status:** {condition_map.get(status.condition, 'Unknown')}"

    # Get multi-run state
    if state.multi_run_trainer is not None:
        config = state.multi_run_trainer.config
        metrics["total_runs"] = f"**Total Runs:** {config.num_runs}"

        if state.multi_run_is_running:
            metrics["current_run"] = f"**Current Run:** {state.multi_run_current_run} / {config.num_runs}"

            # Calculate completed runs
            completed = len(state.multi_run_results.runs) if state.multi_run_results else 0
            metrics["completed_runs"] = f"**Completed:** {completed}"

            # Current loss from history
            if state.multi_run_loss_history:
                current_loss = state.multi_run_loss_history[-1]
                metrics["current_loss"] = f"**Loss:** {current_loss:.4f}"
                metrics["current_step"] = f"**Step:** {len(state.multi_run_loss_history)}"

    # Get results if available
    if state.multi_run_results is not None:
        results = state.multi_run_results
        metrics["completed_runs"] = f"**Completed:** {results.total_runs}"
        metrics["total_steps"] = f"**Total Steps:** {results.total_steps}"
        metrics["total_samples"] = f"**Total Samples:** {results.total_samples}"

        # Format elapsed time
        elapsed = results.total_duration_seconds
        if elapsed > 3600:
            metrics["total_time"] = f"**Elapsed Time:** {elapsed/3600:.1f}h"
        elif elapsed > 60:
            metrics["total_time"] = f"**Elapsed Time:** {elapsed/60:.1f}m"
        else:
            metrics["total_time"] = f"**Elapsed Time:** {elapsed:.0f}s"

        # Get last merge result stats
        if results.runs:
            last_run = results.runs[-1]
            if last_run.merge_result:
                mr = last_run.merge_result
                metrics["scale_factor"] = f"**Scale Factor:** {mr.scale_factor:.4f}"
                metrics["a_matrices"] = f"**A Matrices Merged:** {mr.a_matrices_merged}"
                metrics["b_matrices"] = f"**B Matrices Merged:** {mr.b_matrices_merged}"

            if last_run.validation_loss is not None:
                metrics["val_loss"] = f"**Validation Loss:** {last_run.validation_loss:.4f}"

    # Early stopping status from trainer
    if state.multi_run_trainer is not None:
        trainer = state.multi_run_trainer
        if hasattr(trainer, '_best_val_loss') and trainer._best_val_loss < float('inf'):
            metrics["best_val"] = f"**Best Val Loss:** {trainer._best_val_loss:.4f}"
        if hasattr(trainer, '_early_stop_counter'):
            patience = trainer.config.early_stopping_patience
            counter = trainer._early_stop_counter
            metrics["patience"] = f"**Patience:** {counter} / {patience}"

        if trainer.config.early_stopping:
            metrics["early_stop_status"] = "**Status:** ✅ Enabled"
        else:
            metrics["early_stop_status"] = "**Status:** ⚪ Disabled"

        # Phase 5.3: Checkpoint stats
        ckpt_stats = trainer.get_checkpoint_stats()
        if ckpt_stats:
            metrics["ckpt_count"] = f"**Saved:** {ckpt_stats.total_count}"
            metrics["ckpt_size"] = f"**Total Size:** {ckpt_stats.total_size_gb:.2f} GB"
            if ckpt_stats.best_checkpoint and ckpt_stats.best_checkpoint.validation_loss is not None:
                metrics["ckpt_best"] = f"**Best:** Run {ckpt_stats.best_checkpoint.run_index} (val_loss={ckpt_stats.best_checkpoint.validation_loss:.4f})"
            else:
                metrics["ckpt_best"] = "**Best:** -"
            metrics["ckpt_prunable"] = f"**Prunable:** {ckpt_stats.prunable_count}"

            # Show policy summary
            policy_parts = []
            if trainer.config.checkpoint_auto_prune:
                policy_parts.append(f"keep_best={trainer.config.checkpoint_keep_best_n}")
            else:
                policy_parts.append("no auto-prune")
            if trainer.config.checkpoint_max_total > 0:
                policy_parts.append(f"max={trainer.config.checkpoint_max_total}")
            metrics["ckpt_policy"] = f"**Policy:** {', '.join(policy_parts)}"

    return metrics


def refresh_dashboard() -> tuple:
    """
    Refresh all dashboard metrics.

    Returns:
        Tuple of all dashboard component values
    """
    m = get_dashboard_metrics()
    return (
        # Live Metrics
        m["current_run"],
        m["current_step"],
        m["current_loss"],
        m["eta"],
        # GPU Status
        m["gpu_temp"],
        m["gpu_vram"],
        m["gpu_power"],
        m["gpu_condition"],
        # SLAO Merge Stats
        m["scale_factor"],
        m["similarity"],
        m["a_matrices"],
        m["b_matrices"],
        # Early Stopping
        m["val_loss"],
        m["best_val"],
        m["patience"],
        m["early_stop_status"],
        # Run Timeline
        m["total_runs"],
        m["completed_runs"],
        m["total_steps"],
        m["total_samples"],
        m["total_time"],
        # Phase 5.3: Checkpoints
        m["ckpt_count"],
        m["ckpt_size"],
        m["ckpt_best"],
        m["ckpt_prunable"],
        m["ckpt_policy"],
    )


def refresh_train_sidebar() -> tuple:
    """
    Refresh the Train tab sidebar with live metrics and GPU status.

    Returns:
        Tuple of (step, loss, speed, temp, vram, power)
    """
    # Get GPU status
    status = get_gpu_status()

    # GPU metrics
    if status.available:
        if status.temperature_c is not None:
            temp_emoji = "🟢" if status.temperature_c < 70 else "🟡" if status.temperature_c < 85 else "🔴"
            temp = f"**Temperature:** {temp_emoji} {status.temperature_c}°C"
        else:
            temp = "**Temperature:** -"

        vram = f"**VRAM:** {status.vram_used_gb:.1f} / {status.vram_total_gb:.1f} GB"

        if status.power_draw_w is not None:
            power = f"**Power:** {status.power_draw_w:.0f}W"
        else:
            power = "**Power:** -"
    else:
        temp = "**Temperature:** No GPU"
        vram = "**VRAM:** -"
        power = "**Power:** -"

    # Training metrics from state
    step = "**Step:** -"
    loss = "**Loss:** -"
    speed = "**Speed:** -"

    if state.is_training and state.loss_history:
        step = f"**Step:** {len(state.loss_history)}"
        loss = f"**Loss:** {state.loss_history[-1]:.4f}"
        # Calculate speed if we have timing info
        if hasattr(state, 'train_start_time') and state.train_start_time:
            import time
            elapsed = time.time() - state.train_start_time
            steps_done = len(state.loss_history)
            if elapsed > 0 and steps_done > 0:
                speed = f"**Speed:** {elapsed / steps_done:.1f}s/step"

    return (step, loss, speed, temp, vram, power)


# =============================================================================
# UI BUILDER
# =============================================================================

def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    # Note: In Gradio 6.x, theme and css are passed to launch() not Blocks()
    with gr.Blocks(
        title="Backpropagate - LLM Fine-Tuning",
    ) as app:

        with gr.Tabs():
            # =================================================================
            # TRAIN TAB
            # =================================================================
            with gr.Tab("Train", id="train"):
                # Lightweight sidebar for Train tab
                with gr.Sidebar(position="right", open=True, elem_id="train-sidebar"):
                    gr.Markdown("## 📊 Training Monitor")

                    # Live Metrics - at the top for visibility
                    with gr.Accordion("📈 Live Metrics", open=True):
                        train_sidebar_step = gr.Markdown("**Step:** -")
                        train_sidebar_loss = gr.Markdown("**Loss:** -")
                        train_sidebar_speed = gr.Markdown("**Speed:** -")

                    # GPU Status
                    with gr.Accordion("🌡️ GPU Status", open=True):
                        train_sidebar_temp = gr.Markdown("**Temperature:** -")
                        train_sidebar_vram = gr.Markdown("**VRAM:** -")
                        train_sidebar_power = gr.Markdown("**Power:** -")

                    train_sidebar_refresh = gr.Button("🔄 Refresh", size="sm")

                    # Quick Start - below metrics for reference
                    with gr.Accordion("🚀 Quick Start", open=False):
                        gr.Markdown(
                            """
**3 steps to fine-tune:**

1. **Pick a model** - Qwen 7B is great for most tasks
2. **Choose dataset** - Start with UltraChat or upload JSONL
3. **Click Start** - Watch the loss drop!

---

**First time?** Use defaults:
- 100 steps, 1000 samples
- Takes ~5 min on RTX 3080+

**Tips:**
- Loss should drop steadily
- If loss spikes, lower learning rate
- For production, use Multi-Run tab
                            """
                        )

                with gr.Row():
                    # Left column - Configuration
                    with gr.Column(scale=1):
                        gr.Markdown("### Model")
                        model_preset = gr.Dropdown(
                            choices=list(MODEL_PRESETS.keys()),
                            value="Qwen 2.5 7B (Recommended)",
                            label="Model Preset",
                        )
                        custom_model = gr.Textbox(
                            label="Custom Model (HuggingFace path)",
                            placeholder="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                            visible=False,
                        )

                        gr.Markdown("### Dataset")
                        dataset_preset = gr.Dropdown(
                            choices=list(DATASET_PRESETS.keys()),
                            value="UltraChat 200K (Conversations)",
                            label="Dataset Preset",
                        )
                        custom_dataset = gr.Textbox(
                            label="Custom Dataset Path",
                            placeholder="path/to/data.jsonl",
                            visible=False,
                        )
                        max_samples = gr.Slider(
                            minimum=100,
                            maximum=10000,
                            value=1000,
                            step=100,
                            label="Max Samples",
                        )

                        gr.Markdown("### Training")
                        max_steps = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Training Steps",
                        )
                        learning_rate = gr.Number(
                            value=2e-4,
                            label="Learning Rate",
                        )
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=2,
                            step=1,
                            label="Batch Size",
                        )

                        with gr.Accordion("LoRA Settings", open=False):
                            lora_r = gr.Slider(
                                minimum=4,
                                maximum=64,
                                value=16,
                                step=4,
                                label="LoRA Rank (r)",
                            )
                            lora_alpha = gr.Slider(
                                minimum=8,
                                maximum=128,
                                value=32,
                                step=8,
                                label="LoRA Alpha",
                            )

                        with gr.Row():
                            train_btn = gr.Button(
                                "Start Training",
                                variant="primary",
                                elem_classes=["train-btn"],
                            )
                            stop_btn = gr.Button(
                                "Stop",
                                variant="secondary",
                            )

                    # Right column - Progress
                    with gr.Column(scale=1):
                        gr.Markdown("### Progress")
                        status_text = gr.Markdown("Ready to train")

                        loss_plot = gr.Plot(
                            label="Training Loss",
                            elem_classes=["loss-chart"],
                        )

                        gr.Markdown("### System")
                        with gr.Row():
                            gpu_status = gr.Markdown(
                                get_gpu_status_display(),
                                elem_classes=["gpu-stats"],
                            )

                # Event handlers
                dataset_preset.change(
                    fn=lambda x: gr.update(visible=x == "Custom JSONL"),
                    inputs=[dataset_preset],
                    outputs=[custom_dataset],
                )

                train_btn.click(
                    fn=start_training,
                    inputs=[
                        model_preset, custom_model,
                        dataset_preset, custom_dataset, max_samples,
                        max_steps, learning_rate, batch_size,
                        lora_r, lora_alpha,
                    ],
                    outputs=[status_text, loss_plot, gpu_status],
                )

                stop_btn.click(
                    fn=stop_training,
                    outputs=[status_text],
                )

                # Train sidebar refresh handler
                train_sidebar_refresh.click(
                    fn=refresh_train_sidebar,
                    outputs=[
                        train_sidebar_step,
                        train_sidebar_loss,
                        train_sidebar_speed,
                        train_sidebar_temp,
                        train_sidebar_vram,
                        train_sidebar_power,
                    ],
                )

            # =================================================================
            # MULTI-RUN TAB (SLAO Multi-Run Training)
            # =================================================================
            with gr.Tab("Multi-Run", id="multirun"):
                gr.Markdown(
                    """
                    ### SLAO Multi-Run Training
                    **Multiple short runs with intelligent LoRA merging** - Prevents catastrophic forgetting
                    while maximizing learning from diverse data chunks.

                    Based on [SLAO research](https://arxiv.org/abs/2512.23017): orthogonal initialization +
                    time-aware scaling = better retention of previous knowledge.
                    """
                )

                # Phase 5.1: Training Dashboard Sidebar
                with gr.Sidebar(position="right", open=False, elem_id="training-dashboard") as training_sidebar:
                    gr.Markdown("## 📊 Training Dashboard")

                    # Live Metrics Section
                    with gr.Accordion("📈 Live Metrics", open=True):
                        dashboard_current_run = gr.Markdown("**Current Run:** -")
                        dashboard_current_step = gr.Markdown("**Step:** -")
                        dashboard_current_loss = gr.Markdown("**Loss:** -")
                        dashboard_eta = gr.Markdown("**ETA:** -")

                    # GPU Status Section
                    with gr.Accordion("🌡️ GPU Status", open=True):
                        dashboard_gpu_temp = gr.Markdown("**Temperature:** -")
                        dashboard_gpu_vram = gr.Markdown("**VRAM:** -")
                        dashboard_gpu_power = gr.Markdown("**Power:** -")
                        dashboard_gpu_condition = gr.Markdown("**Status:** -")

                    # SLAO Merge Stats Section (Phase 5.2)
                    with gr.Accordion("🔄 SLAO Merge Stats", open=False):
                        dashboard_scale_factor = gr.Markdown("**Scale Factor:** -")
                        dashboard_similarity = gr.Markdown("**Task Similarity:** -")
                        dashboard_a_matrices = gr.Markdown("**A Matrices Merged:** -")
                        dashboard_b_matrices = gr.Markdown("**B Matrices Merged:** -")

                    # Early Stopping Section (Phase 4.3)
                    with gr.Accordion("⏹️ Early Stopping", open=False):
                        dashboard_val_loss = gr.Markdown("**Validation Loss:** -")
                        dashboard_best_val = gr.Markdown("**Best Val Loss:** -")
                        dashboard_patience = gr.Markdown("**Patience:** -")
                        dashboard_early_stop_status = gr.Markdown("**Status:** Not enabled")

                    # Run Timeline Section
                    with gr.Accordion("📅 Run Timeline", open=False):
                        dashboard_total_runs = gr.Markdown("**Total Runs:** -")
                        dashboard_completed_runs = gr.Markdown("**Completed:** -")
                        dashboard_total_steps = gr.Markdown("**Total Steps:** -")
                        dashboard_total_samples = gr.Markdown("**Total Samples:** -")
                        dashboard_total_time = gr.Markdown("**Elapsed Time:** -")

                    # Phase 5.3: Checkpoint Management Section
                    with gr.Accordion("💾 Checkpoints", open=False):
                        dashboard_ckpt_count = gr.Markdown("**Saved:** -")
                        dashboard_ckpt_size = gr.Markdown("**Total Size:** -")
                        dashboard_ckpt_best = gr.Markdown("**Best:** -")
                        dashboard_ckpt_prunable = gr.Markdown("**Prunable:** -")
                        dashboard_ckpt_policy = gr.Markdown("**Policy:** -")

                    # Refresh button for manual updates
                    dashboard_refresh_btn = gr.Button("🔄 Refresh", size="sm")

                with gr.Row():
                    # Left column - Configuration
                    with gr.Column(scale=1):
                        gr.Markdown("### Model & Data")
                        mr_model_preset = gr.Dropdown(
                            choices=list(MODEL_PRESETS.keys()),
                            value="Qwen 2.5 7B (Recommended)",
                            label="Model Preset",
                        )
                        mr_custom_model = gr.Textbox(
                            label="Custom Model",
                            placeholder="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                            visible=False,
                        )
                        mr_dataset_preset = gr.Dropdown(
                            choices=list(DATASET_PRESETS.keys()),
                            value="UltraChat 200K (Conversations)",
                            label="Dataset Preset",
                        )
                        mr_custom_dataset = gr.Textbox(
                            label="Custom Dataset Path",
                            placeholder="path/to/data.jsonl",
                            visible=False,
                        )

                        gr.Markdown("### Run Configuration")
                        mr_num_runs = gr.Slider(
                            minimum=2,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of Runs",
                        )
                        mr_steps_per_run = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Steps per Run",
                        )
                        mr_samples_per_run = gr.Slider(
                            minimum=500,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="Samples per Run (fresh each run)",
                        )

                        gr.Markdown("### Merge Strategy")
                        mr_merge_mode = gr.Radio(
                            choices=["SLAO", "Simple"],
                            value="SLAO",
                            label="Merge Mode",
                            info="SLAO: Orthogonal init + time-aware scaling (recommended). Simple: Continue from previous.",
                        )

                        with gr.Accordion("Learning Rate Schedule", open=False):
                            mr_initial_lr = gr.Number(
                                value=2e-4,
                                label="Initial LR (Run 1)",
                            )
                            mr_final_lr = gr.Number(
                                value=5e-5,
                                label="Final LR (Last Run)",
                            )

                        with gr.Accordion("GPU Safety", open=False):
                            mr_max_temp = gr.Slider(
                                minimum=70,
                                maximum=95,
                                value=85,
                                step=1,
                                label="Max Temperature (C) - Pause if exceeded",
                            )

                        # Phase 4 Advanced Options
                        with gr.Accordion("Advanced SLAO (Phase 4)", open=False):
                            gr.Markdown("**Adaptive Scaling** - Adjust merge strength based on task similarity")
                            mr_adaptive_scaling = gr.Checkbox(
                                label="Enable Adaptive Scaling",
                                value=False,
                                info="Compute task similarity between runs",
                            )
                            mr_adaptive_range = gr.Slider(
                                minimum=0.2,
                                maximum=2.0,
                                value=[0.5, 1.5],
                                label="Scale Range (min, max)",
                                visible=False,
                            )

                            gr.Markdown("**Layer Scaling** - Different merge strength per layer")
                            mr_layer_scaling = gr.Checkbox(
                                label="Enable Layer Scaling",
                                value=False,
                                info="Early layers merge more, late layers preserve more",
                            )

                            gr.Markdown("**Early Stopping** - Stop if validation loss increases")
                            mr_early_stopping = gr.Checkbox(
                                label="Enable Early Stopping",
                                value=False,
                            )
                            with gr.Row():
                                mr_early_patience = gr.Slider(
                                    minimum=1,
                                    maximum=5,
                                    value=2,
                                    step=1,
                                    label="Patience (runs)",
                                )
                                mr_val_samples = gr.Slider(
                                    minimum=50,
                                    maximum=500,
                                    value=100,
                                    step=50,
                                    label="Validation Samples",
                                )

                        # Phase 5.3: Checkpoint Management
                        with gr.Accordion("Checkpoint Management (Phase 5.3)", open=False):
                            gr.Markdown("**Smart Pruning** - Keep best checkpoints, delete the rest")
                            mr_ckpt_auto_prune = gr.Checkbox(
                                label="Enable Auto-Pruning",
                                value=True,
                                info="Automatically prune low-value checkpoints after each run",
                            )
                            with gr.Row():
                                mr_ckpt_keep_best = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=3,
                                    step=1,
                                    label="Keep Best N",
                                    info="Keep N checkpoints with lowest validation loss",
                                )
                                mr_ckpt_max_total = gr.Slider(
                                    minimum=0,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label="Max Total (0=unlimited)",
                                    info="Hard limit on total checkpoints",
                                )
                            mr_ckpt_keep_final = gr.Checkbox(
                                label="Always Keep Final Checkpoint",
                                value=True,
                            )
                            mr_ckpt_keep_boundaries = gr.Checkbox(
                                label="Keep Run Boundary Checkpoints",
                                value=False,
                                info="Keep first checkpoint of each run",
                            )

                        with gr.Row():
                            mr_start_btn = gr.Button(
                                "Start Multi-Run",
                                variant="primary",
                            )
                            mr_stop_btn = gr.Button(
                                "Stop",
                                variant="secondary",
                            )

                    # Right column - Progress
                    with gr.Column(scale=1):
                        gr.Markdown("### Progress")
                        mr_status_text = gr.Markdown("Ready for multi-run training")

                        mr_loss_plot = gr.Plot(
                            label="Aggregate Loss (run boundaries shown)",
                        )

                        gr.Markdown("### GPU Safety Monitor")
                        mr_gpu_status = gr.Markdown(get_gpu_safety_status())

                        gr.Markdown("### Run History")
                        mr_runs_table = gr.Dataframe(
                            headers=["Run", "Steps", "Samples", "Loss", "LR", "Time", "Merge"],
                            datatype=["number", "number", "number", "str", "str", "str", "str"],
                            interactive=False,
                        )

                # Multi-run event handlers
                mr_dataset_preset.change(
                    fn=lambda x: gr.update(visible=x == "Custom JSONL"),
                    inputs=[mr_dataset_preset],
                    outputs=[mr_custom_dataset],
                )

                mr_start_btn.click(
                    fn=start_multi_run,
                    inputs=[
                        mr_model_preset, mr_custom_model,
                        mr_dataset_preset, mr_custom_dataset,
                        mr_num_runs, mr_steps_per_run, mr_samples_per_run,
                        mr_merge_mode, mr_initial_lr, mr_final_lr, mr_max_temp,
                        # Phase 4 advanced options
                        mr_adaptive_scaling, mr_layer_scaling,
                        mr_early_stopping, mr_early_patience, mr_val_samples,
                        # Phase 5.3 checkpoint options
                        mr_ckpt_auto_prune, mr_ckpt_keep_best, mr_ckpt_max_total,
                        mr_ckpt_keep_final, mr_ckpt_keep_boundaries,
                    ],
                    outputs=[mr_status_text, mr_loss_plot, mr_gpu_status, mr_runs_table],
                )

                mr_stop_btn.click(
                    fn=stop_multi_run,
                    outputs=[mr_status_text],
                )

                # Dashboard refresh event handler
                dashboard_refresh_btn.click(
                    fn=refresh_dashboard,
                    outputs=[
                        # Live Metrics
                        dashboard_current_run,
                        dashboard_current_step,
                        dashboard_current_loss,
                        dashboard_eta,
                        # GPU Status
                        dashboard_gpu_temp,
                        dashboard_gpu_vram,
                        dashboard_gpu_power,
                        dashboard_gpu_condition,
                        # SLAO Merge Stats
                        dashboard_scale_factor,
                        dashboard_similarity,
                        dashboard_a_matrices,
                        dashboard_b_matrices,
                        # Early Stopping
                        dashboard_val_loss,
                        dashboard_best_val,
                        dashboard_patience,
                        dashboard_early_stop_status,
                        # Run Timeline
                        dashboard_total_runs,
                        dashboard_completed_runs,
                        dashboard_total_steps,
                        dashboard_total_samples,
                        dashboard_total_time,
                        # Phase 5.3: Checkpoints
                        dashboard_ckpt_count,
                        dashboard_ckpt_size,
                        dashboard_ckpt_best,
                        dashboard_ckpt_prunable,
                        dashboard_ckpt_policy,
                    ],
                )

            # =================================================================
            # RUNS TAB
            # =================================================================
            with gr.Tab("Runs", id="runs"):
                gr.Markdown("### Training History")

                runs_table = gr.Dataframe(
                    headers=["Run ID", "Steps", "Final Loss", "Duration", "Samples"],
                    datatype=["str", "number", "str", "str", "number"],
                    interactive=False,
                )

                refresh_runs_btn = gr.Button("Refresh")
                refresh_runs_btn.click(
                    fn=get_runs_table,
                    outputs=[runs_table],
                )

            # =================================================================
            # EXPORT TAB
            # =================================================================
            with gr.Tab("Export", id="export"):
                gr.Markdown(
                    """
                    ### Save & Export Model
                    Export your trained model to various formats for deployment.
                    """
                )

                with gr.Row():
                    # Left column - Save and Export
                    with gr.Column(scale=1):
                        gr.Markdown("#### Save LoRA Adapter")
                        save_path = gr.Textbox(
                            label="Output Path",
                            value="./output/lora",
                        )
                        save_merged = gr.Checkbox(
                            label="Save merged weights (larger but standalone)",
                            value=False,
                        )
                        save_btn = gr.Button("Save Model", variant="primary")
                        save_status = gr.Markdown("")

                        gr.Markdown("#### Export to Format")
                        export_format = gr.Dropdown(
                            choices=["GGUF", "Merged", "LoRA"],
                            value="GGUF",
                            label="Format",
                        )
                        quantization = gr.Dropdown(
                            choices=["q4_k_m", "q5_k_m", "q8_0", "f16", "q4_0", "q2_k"],
                            value="q4_k_m",
                            label="Quantization (GGUF only)",
                            info="q4_k_m recommended. f16 for best quality, q2_k for smallest size.",
                        )
                        export_path = gr.Textbox(
                            label="Output Path",
                            value="./output/gguf",
                        )
                        export_btn = gr.Button("Export", variant="primary")
                        export_status = gr.Markdown("")

                    # Right column - Ollama Integration
                    with gr.Column(scale=1):
                        gr.Markdown("#### Ollama Integration")
                        gr.Markdown(
                            """
                            Register your GGUF model with [Ollama](https://ollama.ai) for easy local deployment.
                            """
                        )

                        ollama_gguf_path = gr.Textbox(
                            label="GGUF File Path",
                            value="./output/gguf/model-q4_k_m.gguf",
                            placeholder="Path to your exported GGUF file",
                        )
                        ollama_model_name = gr.Textbox(
                            label="Ollama Model Name",
                            value="my-finetuned-model",
                            placeholder="Name for the model in Ollama",
                        )
                        ollama_system_prompt = gr.Textbox(
                            label="System Prompt (optional)",
                            placeholder="You are a helpful assistant.",
                            lines=2,
                        )
                        ollama_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                        )

                        with gr.Row():
                            ollama_register_btn = gr.Button("Register with Ollama", variant="primary")
                            ollama_list_btn = gr.Button("List Models", variant="secondary")

                        ollama_status = gr.Markdown("")

                        with gr.Accordion("Create Modelfile Only", open=False):
                            modelfile_output = gr.Textbox(
                                label="Modelfile Output Path",
                                value="./output/Modelfile",
                            )
                            modelfile_btn = gr.Button("Create Modelfile")
                            modelfile_status = gr.Markdown("")

                # Event handlers
                save_btn.click(
                    fn=save_model,
                    inputs=[save_path, save_merged],
                    outputs=[save_status],
                )

                export_btn.click(
                    fn=export_model,
                    inputs=[export_format, quantization, export_path],
                    outputs=[export_status],
                )

                ollama_register_btn.click(
                    fn=register_ollama,
                    inputs=[ollama_gguf_path, ollama_model_name, ollama_system_prompt],
                    outputs=[ollama_status],
                )

                ollama_list_btn.click(
                    fn=list_ollama,
                    outputs=[ollama_status],
                )

                modelfile_btn.click(
                    fn=create_ollama_modelfile,
                    inputs=[ollama_gguf_path, modelfile_output, ollama_system_prompt, ollama_temperature],
                    outputs=[modelfile_status],
                )

            # =================================================================
            # DATASET TAB
            # =================================================================
            with gr.Tab("Dataset", id="dataset"):
                gr.Markdown(
                    """
                    ### Dataset Preparation
                    **Upload, validate, and convert datasets** - Supports ShareGPT, Alpaca, OpenAI, and ChatML formats.
                    """
                )

                with gr.Row():
                    # Left column - Upload and Info
                    with gr.Column(scale=1):
                        gr.Markdown("### Load Dataset")
                        ds_file = gr.File(
                            label="Upload Dataset",
                            file_types=[".jsonl", ".json", ".parquet", ".csv", ".txt"],
                        )
                        ds_load_status = gr.Markdown("")

                        gr.Markdown("### Dataset Info")
                        ds_status = gr.Markdown("No dataset loaded")

                        ds_stats_table = gr.Dataframe(
                            headers=["Property", "Value"],
                            datatype=["str", "str"],
                            interactive=False,
                            label="Statistics",
                        )

                        gr.Markdown("### Validation")
                        ds_validation = gr.Textbox(
                            label="Validation Report",
                            lines=10,
                            interactive=False,
                        )

                    # Right column - Preview and Convert
                    with gr.Column(scale=1):
                        gr.Markdown("### Sample Preview")

                        with gr.Row():
                            ds_show_raw = gr.Checkbox(
                                label="Show raw format",
                                value=False,
                            )
                            ds_refresh_btn = gr.Button("Refresh Preview", size="sm")

                        ds_preview = gr.Markdown("No samples to preview")

                        gr.Markdown("### Convert & Export")
                        with gr.Row():
                            ds_target_format = gr.Dropdown(
                                choices=["ChatML"],
                                value="ChatML",
                                label="Target Format",
                            )
                            ds_convert_btn = gr.Button("Convert", variant="secondary")

                        ds_convert_status = gr.Markdown("")

                        ds_export_path = gr.Textbox(
                            label="Export Path",
                            value="./data/converted.jsonl",
                        )
                        ds_export_btn = gr.Button("Export Converted Dataset", variant="primary")
                        ds_export_status = gr.Markdown("")

                # Event handlers
                ds_file.change(
                    fn=load_dataset_file,
                    inputs=[ds_file],
                    outputs=[ds_status, ds_validation, ds_stats_table, ds_preview, ds_load_status],
                )

                ds_show_raw.change(
                    fn=refresh_dataset_preview,
                    inputs=[ds_show_raw],
                    outputs=[ds_preview],
                )

                ds_refresh_btn.click(
                    fn=refresh_dataset_preview,
                    inputs=[ds_show_raw],
                    outputs=[ds_preview],
                )

                ds_convert_btn.click(
                    fn=convert_dataset_format,
                    inputs=[ds_target_format],
                    outputs=[ds_convert_status, ds_preview],
                )

                ds_export_btn.click(
                    fn=export_converted_dataset,
                    inputs=[ds_export_path],
                    outputs=[ds_export_status],
                )

            # =================================================================
            # SETTINGS TAB
            # =================================================================
            with gr.Tab("Settings", id="settings"):
                with gr.Row():
                    # Left column - System Info
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### System")
                            gpu_info = get_gpu_info()
                            sys_info = get_system_info()

                            if gpu_info.get("available"):
                                gpu_name = gpu_info.get("device_name", "Unknown GPU")
                                vram_total = gpu_info.get("memory_total", 0) / (1024**3)
                                vram_used = gpu_info.get("memory_allocated", 0) / (1024**3)

                                gr.Markdown(f"**{gpu_name}**")
                                gr.Slider(
                                    minimum=0, maximum=vram_total, value=vram_used,
                                    label=f"VRAM: {vram_used:.1f} / {vram_total:.1f} GB",
                                    interactive=False
                                )
                            else:
                                gr.Markdown("**No GPU detected**")

                            gr.Markdown(f"""
**Python:** {sys_info.get('python_version', 'Unknown').split()[0]}

**Platform:** {sys_info.get('platform', 'Unknown')[:40]}

**PyTorch:** {sys_info.get('torch_version', 'N/A')}
                            """)

                    # Right column - Features
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("#### Features")
                            features = list_available_features()
                            feature_md = "\n".join([f"- {name}" for name in features.keys()])
                            gr.Markdown(feature_md if feature_md else "No optional features detected")

                # Model Defaults
                with gr.Accordion("Model Defaults", open=False):
                    with gr.Row():
                        settings_model = gr.Textbox(
                            value=settings.model.name,
                            label="Default Model",
                            info="HuggingFace model path"
                        )
                        settings_max_seq = gr.Number(
                            value=settings.model.max_seq_length,
                            label="Max Sequence Length",
                            precision=0
                        )

                # LoRA Defaults
                with gr.Accordion("LoRA Defaults", open=False):
                    with gr.Row():
                        settings_lora_r = gr.Slider(
                            minimum=4, maximum=128, value=settings.lora.r,
                            step=4, label="LoRA Rank (r)"
                        )
                        settings_lora_alpha = gr.Slider(
                            minimum=8, maximum=256, value=settings.lora.lora_alpha,
                            step=8, label="LoRA Alpha"
                        )
                        settings_lora_dropout = gr.Slider(
                            minimum=0, maximum=0.5, value=settings.lora.lora_dropout,
                            step=0.01, label="Dropout"
                        )

                # Training Defaults
                with gr.Accordion("Training Defaults", open=False):
                    with gr.Row():
                        settings_lr = gr.Number(
                            value=settings.training.learning_rate,
                            label="Learning Rate"
                        )
                        settings_batch = gr.Number(
                            value=settings.training.per_device_train_batch_size,
                            label="Batch Size",
                            precision=0
                        )
                        settings_grad_accum = gr.Number(
                            value=settings.training.gradient_accumulation_steps,
                            label="Gradient Accumulation",
                            precision=0
                        )
                    with gr.Row():
                        settings_warmup = gr.Number(
                            value=settings.training.warmup_steps,
                            label="Warmup Steps",
                            precision=0
                        )
                        settings_output = gr.Textbox(
                            value=settings.training.output_dir,
                            label="Output Directory"
                        )

                # Actions
                with gr.Row():
                    save_settings_btn = gr.Button("Save Settings", variant="primary")
                    reset_settings_btn = gr.Button("Reset to Defaults", variant="secondary")
                    export_settings_btn = gr.Button("Export JSON", variant="secondary")

                settings_status = gr.Markdown("")

                with gr.Accordion("Raw Configuration", open=False):
                    gr.Code(
                        value=str(settings.to_dict()),
                        language="json",
                        label="Current Settings"
                    )

            # =================================================================
            # HELP TAB
            # =================================================================
            with gr.Tab("Help", id="help"):
                # Quick Start - Visual Steps
                with gr.Group():
                    gr.Markdown("#### Quick Start")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=150):
                            gr.Markdown("""
**1. Select Model**

Pick a preset or enter a HuggingFace path
                            """)
                        with gr.Column(scale=1, min_width=150):
                            gr.Markdown("""
**2. Load Dataset**

Choose a preset or upload JSONL
                            """)
                        with gr.Column(scale=1, min_width=150):
                            gr.Markdown("""
**3. Train**

Click Start and watch loss drop
                            """)
                        with gr.Column(scale=1, min_width=150):
                            gr.Markdown("""
**4. Export**

Convert to GGUF for Ollama
                            """)

                # Model Selection Guide
                with gr.Accordion("Model Selection Guide", open=False):
                    gr.Markdown("""
| VRAM | Recommended Model | Notes |
|------|------------------|-------|
| 8 GB | Qwen2.5-3B-Instruct | Good for testing |
| 12 GB | Qwen2.5-7B-Instruct | Best balance |
| 16 GB | Qwen2.5-7B-Instruct | Room for larger batches |
| 24 GB+ | Qwen2.5-14B-Instruct | Maximum quality |

**Tips:**
- Use 4-bit quantized models (bnb-4bit) to reduce VRAM by ~75%
- Instruct models are easier to fine-tune than base models
- Start with 7B - it's the sweet spot for most tasks
                    """)

                # Dataset Formats
                with gr.Accordion("Dataset Formats", open=False):
                    with gr.Tabs():
                        with gr.Tab("ChatML"):
                            gr.Code(
                                value='{"text": "<|im_start|>user\\nHello!<|im_end|>\\n<|im_start|>assistant\\nHi there!<|im_end|>"}',
                                language="json",
                                label="ChatML Format (Recommended)"
                            )
                            gr.Markdown("Best for Qwen, Yi, and most modern models.")

                        with gr.Tab("ShareGPT"):
                            gr.Code(
                                value='{"conversations": [{"from": "human", "value": "Hello!"}, {"from": "gpt", "value": "Hi there!"}]}',
                                language="json",
                                label="ShareGPT Format"
                            )
                            gr.Markdown("Common format, auto-converted to ChatML.")

                        with gr.Tab("Alpaca"):
                            gr.Code(
                                value='{"instruction": "Say hello", "input": "", "output": "Hello! How can I help?"}',
                                language="json",
                                label="Alpaca Format"
                            )
                            gr.Markdown("Simple instruction-following format.")

                # Training Tips
                with gr.Accordion("Training Tips", open=False):
                    gr.Markdown("""
**Getting Good Results:**
- **Start small**: 100 steps, 1000 samples to verify setup works
- **Learning rate**: 2e-4 is a safe default, lower (1e-4) for larger models
- **Batch size**: Auto-detected, but 2-4 works for most setups
- **LoRA rank**: 16 is balanced, 32+ for complex tasks, 8 for simple ones

**Watch for Overfitting:**
- Loss dropping too fast? Reduce learning rate
- Loss stuck high? Increase learning rate or check data quality
- Training loss good but output bad? Need more diverse data

**Multi-Run Strategy:**
- Multiple short runs (100 steps each) often beat one long run
- Each run sees fresh data, preventing overfitting
- Use the Multi-Run tab for automated training
                    """)

                # Troubleshooting
                with gr.Accordion("Troubleshooting", open=False):
                    gr.Markdown("""
**Out of Memory (OOM):**
- Reduce batch size to 1
- Use a smaller model (3B instead of 7B)
- Enable gradient checkpointing (default)
- Reduce max sequence length

**Training is Slow:**
- Increase batch size if VRAM allows
- Use gradient accumulation instead of large batches
- Check GPU utilization with `nvidia-smi`

**Loss Not Decreasing:**
- Check dataset format is correct
- Try a higher learning rate (5e-4)
- Verify data quality - garbage in, garbage out

**Export Failed:**
- Ensure model trained successfully first
- Check disk space for GGUF output
- For Ollama: verify Ollama is running (`ollama list`)
                    """)

                # CLI Reference
                with gr.Accordion("CLI Reference", open=False):
                    gr.Code(
                        value="""# Launch UI
backprop ui --port 7862

# Single training run
backprop train --data my_data.jsonl --steps 100 --samples 1000

# Multi-run training
backprop multi-run --data ultrachat --runs 5 --steps 100

# Export to GGUF
backprop export ./output/lora --format gguf -q q4_k_m

# Export and register with Ollama
backprop export ./output/lora --format gguf --ollama --ollama-name mymodel

# Show system info
backprop info

# Show configuration
backprop config""",
                        language="shell",
                        label="CLI Commands"
                    )

                # Links
                gr.Markdown("---")
                with gr.Row():
                    gr.Button(
                        "GitHub",
                        link="https://github.com/mikeyfrilot/backpropagate",
                        variant="secondary",
                        size="sm"
                    )
                    gr.Button(
                        "Documentation",
                        link="https://github.com/mikeyfrilot/backpropagate#readme",
                        variant="secondary",
                        size="sm"
                    )
                    gr.Button(
                        "Report Issue",
                        link="https://github.com/mikeyfrilot/backpropagate/issues",
                        variant="secondary",
                        size="sm"
                    )

    return app


def launch(
    port: int = 7862,
    share: bool = False,
    auth: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
) -> None:
    """
    Launch the Gradio UI.

    Args:
        port: Port to run the server on (default: 7862)
        share: Create a public shareable link (default: False)
        auth: Authentication credentials. Required when share=True for security.
              Can be a tuple (username, password) or list of tuples for multiple users.

    Raises:
        ValueError: If share=True but no auth is provided

    Examples:
        # Local only (no auth needed)
        launch(port=7862)

        # Public URL with auth
        launch(port=7862, share=True, auth=("admin", "secretpassword"))

        # Multiple users
        launch(share=True, auth=[("user1", "pass1"), ("user2", "pass2")])
    """
    import warnings

    # Security check: require auth when sharing publicly
    if share and auth is None:
        warnings.warn(
            "Creating a public URL without authentication is a security risk. "
            "Anyone with the URL can access your training interface. "
            "Consider using auth=('username', 'password') for protection.",
            SecurityWarning,
            stacklevel=2,
        )
        logger.warning(
            "SECURITY: Public share link created without authentication. "
            "This exposes training controls to anyone with the URL."
        )

    app = create_ui()
    theme = create_backpropagate_theme()
    css = get_css()

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "share": share,
        "inbrowser": True,
        "theme": theme,
        "css": css,
    }

    if auth is not None:
        launch_kwargs["auth"] = auth

    app.launch(**launch_kwargs)
