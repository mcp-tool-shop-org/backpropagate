"""
Backpropagate - Configuration Management
=========================================

Modern configuration using pydantic-settings for type-safe environment variable parsing.
All settings can be overridden via environment variables with BACKPROPAGATE_ prefix.

Example:
    BACKPROPAGATE_TRAINING__LEARNING_RATE=2e-4
    BACKPROPAGATE_TRAINING__BATCH_SIZE=4
    BACKPROPAGATE_MODEL__NAME=unsloth/Qwen2.5-7B-Instruct-bnb-4bit

Features:
- Type-safe configuration with automatic validation
- Nested config via double underscore delimiter (__)
- .env file support
- Cached settings instance via @lru_cache
- Windows-safe defaults baked in
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional, List
import os

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "reload_settings",
    "get_output_dir",
    "get_cache_dir",
    # Sub-configs
    "ModelConfig",
    "TrainingConfig",
    "LoRAConfig",
    "DataConfig",
    "UIConfig",
    "WindowsConfig",
    "SecurityConfig",
    # Training presets (Phase 1.2)
    "TrainingPreset",
    "TRAINING_PRESETS",
    "get_preset",
    # LR scaling helpers (Phase 1.3)
    "get_recommended_lr",
    "get_recommended_warmup",
    # Constants
    "PYDANTIC_SETTINGS_AVAILABLE",
]

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    PYDANTIC_SETTINGS_AVAILABLE = False


# =============================================================================
# WINDOWS-SAFE DEFAULTS (Based on RTX 5080 testing)
# =============================================================================

WINDOWS_DEFAULTS = {
    "dataloader_num_workers": 0,
    "tokenizers_parallelism": False,
    "xformers_disabled": True,  # SM 12.0+ (Blackwell/Ada)
    "cuda_launch_blocking": True,
    "pre_tokenize": True,  # Avoid multiprocessing crashes
}


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

if PYDANTIC_SETTINGS_AVAILABLE:

    class ModelConfig(BaseSettings):
        """Model configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_MODEL__",
            env_ignore_empty=True,
        )

        # Model identifier (HuggingFace path or local path)
        name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
        # Whether to use 4-bit quantization
        load_in_4bit: bool = True
        # Maximum sequence length
        max_seq_length: int = 2048
        # Data type for training
        dtype: Optional[str] = None  # Auto-detect (bf16 on Ampere+)
        # Trust remote code from HuggingFace
        trust_remote_code: bool = True

    class LoRAConfig(BaseSettings):
        """LoRA/QLoRA configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_LORA__",
            env_ignore_empty=True,
        )

        # LoRA rank (dimension)
        r: int = 16
        # LoRA alpha (scaling factor)
        lora_alpha: int = 32
        # Dropout rate
        lora_dropout: float = 0.05
        # Target modules for LoRA
        target_modules: List[str] = Field(default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        # Use gradient checkpointing (reduces VRAM by ~30%)
        use_gradient_checkpointing: str = "unsloth"  # "unsloth" or True/False
        # Random state for reproducibility
        random_state: int = 42

    class TrainingConfig(BaseSettings):
        """Training hyperparameters."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_TRAINING__",
            env_ignore_empty=True,
        )

        # Batch size per device
        per_device_train_batch_size: int = 2
        # Gradient accumulation steps (effective batch = batch_size * grad_accum)
        gradient_accumulation_steps: int = 4
        # Number of training steps (0 = use num_train_epochs)
        max_steps: int = 100
        # Number of epochs (ignored if max_steps > 0)
        num_train_epochs: int = 1
        # Learning rate
        learning_rate: float = 2e-4
        # Weight decay
        weight_decay: float = 0.01
        # Warmup steps
        warmup_steps: int = 10
        # Warmup ratio (alternative to warmup_steps)
        warmup_ratio: float = 0.0
        # Optimizer
        optim: str = "adamw_8bit"
        # LR scheduler type
        lr_scheduler_type: str = "cosine"
        # Logging steps
        logging_steps: int = 10
        # Save steps
        save_steps: int = 100
        # Use bf16 (recommended for Ampere+)
        bf16: bool = True
        # Use fp16 (for older GPUs)
        fp16: bool = False
        # Seed for reproducibility
        seed: int = 42
        # Output directory
        output_dir: str = "./output"
        # Overwrite output directory
        overwrite_output_dir: bool = True

    class DataConfig(BaseSettings):
        """Dataset configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_DATA__",
            env_ignore_empty=True,
        )

        # Dataset name or path
        dataset_name: str = "HuggingFaceH4/ultrachat_200k"
        # Dataset split
        dataset_split: str = "train_sft"
        # Number of samples (0 = all)
        max_samples: int = 1000
        # Text column name
        text_column: str = "text"
        # Chat template format (chatml, llama, alpaca, sharegpt)
        chat_format: str = "chatml"
        # Pre-tokenize dataset (Windows-safe)
        pre_tokenize: bool = True
        # Shuffle dataset
        shuffle: bool = True
        # Packing (combine short sequences)
        packing: bool = False

    class UIConfig(BaseSettings):
        """Gradio UI configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_UI__",
            env_ignore_empty=True,
        )

        port: int = 7862
        host: str = "127.0.0.1"  # Localhost only for security
        share: bool = False
        auto_open: bool = True

    class WindowsConfig(BaseSettings):
        """Windows-specific settings (auto-applied on Windows)."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_WINDOWS__",
            env_ignore_empty=True,
        )

        # Number of dataloader workers (0 for Windows)
        dataloader_num_workers: int = 0
        # Disable tokenizers parallelism
        tokenizers_parallelism: bool = False
        # Disable xformers (incompatible with SM 12.0+)
        xformers_disabled: bool = True
        # CUDA launch blocking for debugging
        cuda_launch_blocking: bool = False
        # Pre-tokenize to avoid multiprocessing issues
        pre_tokenize: bool = True

    class MultiRunConfig(BaseSettings):
        """Multi-run training configuration."""
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_MULTIRUN__",
            env_ignore_empty=True,
        )

        # Number of training runs
        num_runs: int = 5
        # Steps per run
        steps_per_run: int = 100
        # Samples per run (new samples each run)
        samples_per_run: int = 1000
        # Whether to continue from previous LoRA
        continue_from_previous: bool = True
        # Save intermediate checkpoints
        save_intermediate: bool = True

    class SecurityConfig(BaseSettings):
        """
        Security configuration for production deployments.

        Environment Variables:
            BACKPROPAGATE_SECURITY__REQUIRE_AUTH=true
            BACKPROPAGATE_SECURITY__ALLOWED_PATHS=/data,/models
            BACKPROPAGATE_SECURITY__SESSION_TIMEOUT_MINUTES=30
            BACKPROPAGATE_SECURITY__ENABLE_CSRF=true

        2026 Best Practices:
        - Require auth when share=True (public URLs)
        - Restrict file access to allowed directories
        - Use JWT with short expiry for sessions
        - Enable CSRF protection for state-changing requests
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_SECURITY__",
            env_ignore_empty=True,
        )

        # Authentication
        require_auth: bool = False  # Set True in production
        auth_username: Optional[str] = None
        auth_password: Optional[str] = Field(default=None, json_schema_extra={"secret": True})

        # Path restrictions
        allowed_paths: Optional[List[str]] = None  # None = no restriction
        block_path_traversal: bool = True

        # Session management
        session_timeout_minutes: int = 30
        jwt_secret: Optional[str] = Field(default=None, json_schema_extra={"secret": True})
        jwt_algorithm: str = "HS256"

        # CSRF protection
        enable_csrf: bool = True
        csrf_token_expiry_minutes: int = 60

        # Rate limiting
        rate_limit_training: int = 3  # Max training starts per minute
        rate_limit_export: int = 5    # Max exports per minute

        # Logging
        audit_log_enabled: bool = True
        audit_log_file: Optional[str] = None  # None = stdout only

        # Content Security Policy
        enable_csp: bool = True
        csp_report_only: bool = False  # Set False to enforce

        def get_auth_tuple(self) -> Optional[tuple]:
            """Get auth tuple for Gradio if credentials are set."""
            if self.auth_username and self.auth_password:
                return (self.auth_username, self.auth_password)
            return None

        def validate_production_config(self) -> List[str]:
            """Check for security misconfigurations. Returns list of warnings."""
            warnings = []
            if not self.require_auth:
                warnings.append("SECURITY: require_auth is False - UI is unprotected")
            if not self.jwt_secret:
                warnings.append("SECURITY: jwt_secret not set - using random secret (sessions lost on restart)")
            if not self.enable_csrf:
                warnings.append("SECURITY: CSRF protection disabled")
            if self.session_timeout_minutes > 480:  # 8 hours
                warnings.append("SECURITY: session_timeout_minutes > 8 hours - consider shorter timeout")
            return warnings

    class Settings(BaseSettings):
        """
        Main settings container using pydantic-settings.

        All settings are loaded from environment variables with BACKPROPAGATE_ prefix.
        Nested settings use double underscore (__) as delimiter.

        Usage:
            from backpropagate.config import get_settings

            settings = get_settings()
            print(settings.model.name)
            print(settings.training.learning_rate)

        Environment Examples:
            BACKPROPAGATE_MODEL__NAME=unsloth/Llama-3.2-3B-Instruct-bnb-4bit
            BACKPROPAGATE_TRAINING__LEARNING_RATE=1e-4
            BACKPROPAGATE_LORA__R=32
        """
        model_config = SettingsConfigDict(
            env_prefix="BACKPROPAGATE_",
            env_file=".env",
            env_file_encoding="utf-8",
            env_ignore_empty=True,
            env_nested_delimiter="__",
            extra="ignore",
        )

        # Nested configs
        model: ModelConfig = Field(default_factory=ModelConfig)
        training: TrainingConfig = Field(default_factory=TrainingConfig)
        lora: LoRAConfig = Field(default_factory=LoRAConfig)
        data: DataConfig = Field(default_factory=DataConfig)
        ui: UIConfig = Field(default_factory=UIConfig)
        windows: WindowsConfig = Field(default_factory=WindowsConfig)
        multi_run: MultiRunConfig = Field(default_factory=MultiRunConfig)
        security: SecurityConfig = Field(default_factory=SecurityConfig)

        # Package info
        version: str = "0.1.0"
        name: str = "backpropagate"

        def to_dict(self) -> dict:
            """Export settings as dictionary."""
            return {
                "version": self.version,
                "model": {
                    "name": self.model.name,
                    "load_in_4bit": self.model.load_in_4bit,
                    "max_seq_length": self.model.max_seq_length,
                },
                "training": {
                    "batch_size": self.training.per_device_train_batch_size,
                    "grad_accum": self.training.gradient_accumulation_steps,
                    "learning_rate": self.training.learning_rate,
                    "max_steps": self.training.max_steps,
                },
                "lora": {
                    "r": self.lora.r,
                    "alpha": self.lora.lora_alpha,
                    "dropout": self.lora.lora_dropout,
                },
                "data": {
                    "dataset": self.data.dataset_name,
                    "max_samples": self.data.max_samples,
                },
            }

        def apply_windows_fixes(self) -> None:
            """Apply Windows-specific environment variables."""
            if os.name == "nt":  # Windows
                os.environ["TOKENIZERS_PARALLELISM"] = str(self.windows.tokenizers_parallelism).lower()
                if self.windows.xformers_disabled:
                    os.environ["XFORMERS_DISABLED"] = "1"
                if self.windows.cuda_launch_blocking:
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

else:
    # Fallback implementation using dataclasses
    from dataclasses import dataclass, field

    def _get_env(key: str, default: str = None) -> Optional[str]:
        return os.environ.get(f"BACKPROPAGATE_{key}", default)

    def _get_env_int(key: str, default: int) -> int:
        val = _get_env(key)
        return int(val) if val else default

    def _get_env_float(key: str, default: float) -> float:
        val = _get_env(key)
        return float(val) if val else default

    def _get_env_bool(key: str, default: bool) -> bool:
        val = _get_env(key)
        return val.lower() in ("true", "1", "yes") if val else default

    @dataclass
    class ModelConfig:
        name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
        load_in_4bit: bool = True
        max_seq_length: int = 2048
        dtype: Optional[str] = None
        trust_remote_code: bool = True

    @dataclass
    class LoRAConfig:
        r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        target_modules: List[str] = field(default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
        use_gradient_checkpointing: str = "unsloth"
        random_state: int = 42

    @dataclass
    class TrainingConfig:
        per_device_train_batch_size: int = 2
        gradient_accumulation_steps: int = 4
        max_steps: int = 100
        num_train_epochs: int = 1
        learning_rate: float = 2e-4
        weight_decay: float = 0.01
        warmup_steps: int = 10
        warmup_ratio: float = 0.0
        optim: str = "adamw_8bit"
        lr_scheduler_type: str = "cosine"
        logging_steps: int = 10
        save_steps: int = 100
        bf16: bool = True
        fp16: bool = False
        seed: int = 42
        output_dir: str = "./output"
        overwrite_output_dir: bool = True

    @dataclass
    class DataConfig:
        dataset_name: str = "HuggingFaceH4/ultrachat_200k"
        dataset_split: str = "train_sft"
        max_samples: int = 1000
        text_column: str = "text"
        chat_format: str = "chatml"
        pre_tokenize: bool = True
        shuffle: bool = True
        packing: bool = False

    @dataclass
    class UIConfig:
        port: int = 7862
        host: str = "127.0.0.1"
        share: bool = False
        auto_open: bool = True

    @dataclass
    class WindowsConfig:
        dataloader_num_workers: int = 0
        tokenizers_parallelism: bool = False
        xformers_disabled: bool = True
        cuda_launch_blocking: bool = False
        pre_tokenize: bool = True

    @dataclass
    class MultiRunConfig:
        num_runs: int = 5
        steps_per_run: int = 100
        samples_per_run: int = 1000
        continue_from_previous: bool = True
        save_intermediate: bool = True

    @dataclass
    class SecurityConfig:
        """Security configuration (fallback without pydantic-settings)."""
        require_auth: bool = False
        auth_username: Optional[str] = None
        auth_password: Optional[str] = None
        allowed_paths: Optional[List[str]] = None
        block_path_traversal: bool = True
        session_timeout_minutes: int = 30
        jwt_secret: Optional[str] = None
        jwt_algorithm: str = "HS256"
        enable_csrf: bool = True
        csrf_token_expiry_minutes: int = 60
        rate_limit_training: int = 3
        rate_limit_export: int = 5
        audit_log_enabled: bool = True
        audit_log_file: Optional[str] = None
        enable_csp: bool = True
        csp_report_only: bool = False

        def get_auth_tuple(self) -> Optional[tuple]:
            if self.auth_username and self.auth_password:
                return (self.auth_username, self.auth_password)
            return None

        def validate_production_config(self) -> List[str]:
            warnings = []
            if not self.require_auth:
                warnings.append("SECURITY: require_auth is False")
            if not self.jwt_secret:
                warnings.append("SECURITY: jwt_secret not set")
            return warnings

    @dataclass
    class Settings:
        model: ModelConfig = field(default_factory=ModelConfig)
        training: TrainingConfig = field(default_factory=TrainingConfig)
        lora: LoRAConfig = field(default_factory=LoRAConfig)
        data: DataConfig = field(default_factory=DataConfig)
        ui: UIConfig = field(default_factory=UIConfig)
        windows: WindowsConfig = field(default_factory=WindowsConfig)
        multi_run: MultiRunConfig = field(default_factory=MultiRunConfig)
        security: SecurityConfig = field(default_factory=SecurityConfig)
        version: str = "0.1.0"
        name: str = "backpropagate"

        def to_dict(self) -> dict:
            return {"version": self.version}

        def apply_windows_fixes(self) -> None:
            if os.name == "nt":
                os.environ["TOKENIZERS_PARALLELISM"] = str(self.windows.tokenizers_parallelism).lower()
                if self.windows.xformers_disabled:
                    os.environ["XFORMERS_DISABLED"] = "1"


# =============================================================================
# CACHED SETTINGS INSTANCE
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses @lru_cache to avoid re-reading environment/.env on every call.
    Call get_settings.cache_clear() to reload settings.
    """
    return Settings()


# Backwards-compatible singleton
settings = get_settings()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def reload_settings() -> Settings:
    """Reload all settings from environment variables."""
    get_settings.cache_clear()
    global settings
    settings = get_settings()
    return settings


def get_output_dir() -> Path:
    """Get the output directory for trained models."""
    output_dir = Path(settings.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_cache_dir() -> Path:
    """Get the cache directory for this package."""
    cache_dir = Path.home() / ".cache" / "backpropagate"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_training_args() -> dict:
    """
    Get training arguments as a dict for TrainingArguments.

    Returns:
        Dict compatible with transformers.TrainingArguments
    """
    s = settings
    return {
        "per_device_train_batch_size": s.training.per_device_train_batch_size,
        "gradient_accumulation_steps": s.training.gradient_accumulation_steps,
        "max_steps": s.training.max_steps if s.training.max_steps > 0 else -1,
        "num_train_epochs": s.training.num_train_epochs,
        "learning_rate": s.training.learning_rate,
        "weight_decay": s.training.weight_decay,
        "warmup_steps": s.training.warmup_steps,
        "warmup_ratio": s.training.warmup_ratio,
        "optim": s.training.optim,
        "lr_scheduler_type": s.training.lr_scheduler_type,
        "logging_steps": s.training.logging_steps,
        "save_steps": s.training.save_steps,
        "bf16": s.training.bf16,
        "fp16": s.training.fp16,
        "seed": s.training.seed,
        "output_dir": s.training.output_dir,
        "overwrite_output_dir": s.training.overwrite_output_dir,
        "dataloader_num_workers": s.windows.dataloader_num_workers if os.name == "nt" else 4,
    }


# =============================================================================
# TRAINING PRESETS (Phase 1.2)
# =============================================================================
# Research shows LoRA works best with effective batch size 8-32
# See: https://arxiv.org/abs/2512.23017, Unsloth hyperparameters guide

from dataclasses import dataclass as dc_dataclass

@dc_dataclass
class TrainingPreset:
    """Training configuration preset for different use cases."""
    name: str
    description: str
    # LoRA
    lora_r: int
    lora_alpha: int
    # Batch
    batch_size: int
    gradient_accumulation: int
    # Learning rate
    learning_rate: float
    warmup_steps: int
    # Multi-run
    steps_per_run: int
    num_runs: int
    # Optional
    samples_per_run: int = 1000
    replay_fraction: float = 0.0
    validate_every_run: bool = False

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size (batch_size * gradient_accumulation)."""
        return self.batch_size * self.gradient_accumulation


# Research-backed presets based on SLAO paper, Unsloth docs, and Databricks guide
TRAINING_PRESETS = {
    "fast": TrainingPreset(
        name="fast",
        description="Quick iterations for testing and debugging",
        lora_r=8,
        lora_alpha=16,
        batch_size=2,
        gradient_accumulation=4,  # effective=8
        learning_rate=5e-4,
        warmup_steps=5,
        steps_per_run=50,
        num_runs=3,
        samples_per_run=500,
    ),
    "balanced": TrainingPreset(
        name="balanced",
        description="Default recommended preset for most use cases",
        lora_r=16,
        lora_alpha=32,
        batch_size=2,
        gradient_accumulation=8,  # effective=16
        learning_rate=2e-4,
        warmup_steps=10,
        steps_per_run=100,
        num_runs=5,
        samples_per_run=1000,
    ),
    "quality": TrainingPreset(
        name="quality",
        description="Maximum training effectiveness for final models",
        lora_r=32,
        lora_alpha=64,
        batch_size=4,
        gradient_accumulation=8,  # effective=32
        learning_rate=1e-4,
        warmup_steps=20,
        steps_per_run=200,
        num_runs=10,
        samples_per_run=2000,
        replay_fraction=0.1,
        validate_every_run=True,
    ),
}


def get_preset(name: str) -> TrainingPreset:
    """
    Get a training preset by name.

    Args:
        name: Preset name ("fast", "balanced", or "quality")

    Returns:
        TrainingPreset configuration

    Raises:
        ValueError: If preset name is not recognized

    Example:
        >>> preset = get_preset("balanced")
        >>> trainer = Trainer(
        ...     lora_r=preset.lora_r,
        ...     learning_rate=preset.learning_rate,
        ... )
    """
    if name not in TRAINING_PRESETS:
        available = ", ".join(TRAINING_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return TRAINING_PRESETS[name]


def get_recommended_lr(dataset_size: int, base_lr: float = 2e-4) -> float:
    """
    Get recommended learning rate based on dataset size (Phase 1.3).

    Research shows LoRA benefits from ~10× higher LR than full fine-tuning,
    but should be adjusted based on dataset size to prevent overfitting.

    Args:
        dataset_size: Number of training samples
        base_lr: Base learning rate (default: 2e-4)

    Returns:
        Recommended learning rate

    Reference:
        - Small datasets (<1K): Higher LR with more warmup to learn quickly
        - Medium datasets (1K-10K): Standard LR
        - Large datasets (>10K): Lower LR for stability

    Example:
        >>> lr = get_recommended_lr(500)  # Returns 5e-4
        >>> lr = get_recommended_lr(5000)  # Returns 2e-4
        >>> lr = get_recommended_lr(50000)  # Returns 1e-4
    """
    if dataset_size < 1000:
        # Small dataset: higher LR, more aggressive learning
        return 5e-4
    elif dataset_size < 10000:
        # Medium dataset: standard LoRA LR
        return base_lr
    else:
        # Large dataset: lower LR for stability
        return 1e-4


def get_recommended_warmup(dataset_size: int, num_steps: int) -> int:
    """
    Get recommended warmup steps based on dataset size (Phase 1.3).

    Small datasets need more warmup to prevent early instability.
    Large datasets can use standard warmup ratios.

    Args:
        dataset_size: Number of training samples
        num_steps: Total number of training steps

    Returns:
        Recommended warmup steps

    Example:
        >>> warmup = get_recommended_warmup(500, 100)  # Returns 15 (15%)
        >>> warmup = get_recommended_warmup(5000, 100)  # Returns 10 (10%)
        >>> warmup = get_recommended_warmup(50000, 100)  # Returns 5 (5%)
    """
    if dataset_size < 1000:
        # Small dataset: 15% warmup
        ratio = 0.15
    elif dataset_size < 10000:
        # Medium dataset: 10% warmup
        ratio = 0.10
    else:
        # Large dataset: 5% warmup
        ratio = 0.05

    return max(1, int(num_steps * ratio))
