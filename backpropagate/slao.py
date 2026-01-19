"""
Backpropagate - SLAO (Single LoRA Continual Learning via Asymmetric Merging)
=============================================================================

Implementation of the SLAO method from "Merge before Forget: A Single LoRA
Continual Learning via Continual Merging" (arXiv:2512.23017).

Key innovations:
- Orthogonal initialization via QR decomposition to minimize forgetting
- Asymmetric handling of LoRA A and B matrices
- Time-aware scaling factor lambda(i) = 1/sqrt(i) for balanced merging

Usage:
    from backpropagate.slao import SLAOMerger

    merger = SLAOMerger()

    # After each training run, merge the new LoRA into the accumulated one
    merged_lora = merger.merge(
        previous_lora=lora_run_1,
        new_lora=lora_run_2,
        run_index=2
    )

References:
    - Paper: https://arxiv.org/abs/2512.23017
    - K-Merge: https://arxiv.org/abs/2510.13537
"""

import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import json

from .exceptions import SLAOError, SLAOMergeError, SLAOCheckpointError, InvalidSettingError
from .security import check_torch_security

logger = logging.getLogger(__name__)

__all__ = [
    "SLAOMerger",
    "SLAOConfig",
    "MergeResult",
    "time_aware_scale",
    "orthogonal_init_A",
    "merge_lora_weights",
    # Phase 4 additions
    "compute_task_similarity",
    "adaptive_scale",
    "get_layer_scale",
]

@dataclass
class SLAOConfig:
    """Configuration for SLAO merging."""

    # Use time-aware scaling (recommended)
    use_time_aware_scaling: bool = True

    # Use orthogonal initialization for A matrix (recommended)
    use_orthogonal_init: bool = True

    # Custom scaling function: "sqrt" (1/sqrt(i)), "linear" (1/i), "log", "constant" (1)
    scaling_type: str = "sqrt"

    # Minimum scaling factor (prevents vanishing updates)
    min_scale: float = 0.1

    # Whether to normalize after merging
    normalize_after_merge: bool = False

    # Save merge history for debugging
    save_merge_history: bool = True

    # Phase 4.1: Adaptive scaling based on task similarity
    use_adaptive_scaling: bool = False
    adaptive_scale_range: Tuple[float, float] = (0.5, 1.5)  # Scale multiplier range

    # Phase 4.2: Selective layer merging
    use_layer_scaling: bool = False
    layer_scale_early: float = 0.3    # Layers 0-33% (more aggressive merge)
    layer_scale_middle: float = 0.5   # Layers 33-66%
    layer_scale_late: float = 0.7     # Layers 66-100% (preserve more)


@dataclass
class MergeResult:
    """Result of a SLAO merge operation."""

    run_index: int
    scale_factor: float
    a_matrices_merged: int
    b_matrices_merged: int
    total_params_merged: int
    merge_time_seconds: float

    # Optional diagnostics
    a_norm_before: Optional[float] = None
    a_norm_after: Optional[float] = None
    b_norm_before: Optional[float] = None
    b_norm_after: Optional[float] = ""


# =============================================================================
# CORE SLAO FUNCTIONS
# =============================================================================

def time_aware_scale(run_index: int, scaling_type: str = "sqrt", min_scale: float = 0.1) -> float:
    """
    Compute time-aware scaling factor for SLAO merging.

    From the paper: "lambda(i) = 1/sqrt(i) is a natural choice for the scaling
    factor" because task vectors from different tasks tend to be approximately
    orthogonal.

    Args:
        run_index: Current run index (1-based, first run = 1)
        scaling_type: Type of scaling:
            - "sqrt": 1/√i (paper default, good balance)
            - "linear": 1/i (more aggressive, preserves early learning)
            - "log": 1/log(i+1) (slower decay, more plasticity)
            - "constant": 1.0 (simple EMA, no decay)
        min_scale: Minimum scaling factor to prevent vanishing updates

    Returns:
        Scaling factor in range [min_scale, 1.0]

    Raises:
        InvalidSettingError: If run_index or scaling_type is invalid

    Example:
        >>> time_aware_scale(1, "sqrt")   # 1.0
        >>> time_aware_scale(4, "sqrt")   # 0.5
        >>> time_aware_scale(9, "sqrt")   # 0.333
        >>> time_aware_scale(4, "log")    # 0.621 (slower decay)
    """
    if not isinstance(run_index, int) or run_index < 1:
        raise InvalidSettingError(
            "run_index", run_index, "positive integer >= 1",
            suggestion="Run index should start at 1 for the first run"
        )

    valid_scaling_types = ("sqrt", "linear", "log", "constant")
    if scaling_type not in valid_scaling_types:
        raise InvalidSettingError(
            "scaling_type", scaling_type, f"one of {valid_scaling_types}",
            suggestion="Use 'sqrt' for recommended time-aware scaling"
        )

    if scaling_type == "sqrt":
        # Paper recommendation: lambda(i) = 1/sqrt(i)
        scale = 1.0 / math.sqrt(run_index)
    elif scaling_type == "linear":
        # More aggressive decay: lambda(i) = 1/i
        scale = 1.0 / run_index
    elif scaling_type == "log":
        # Slower decay: lambda(i) = 1/log(i+1)
        # Maintains more plasticity in later runs
        scale = 1.0 / math.log(run_index + 1)
    elif scaling_type == "constant":
        # No decay (simple averaging)
        scale = 1.0

    return max(scale, min_scale)


def orthogonal_init_A(A_prev: "torch.Tensor") -> "torch.Tensor":
    """
    Initialize new A matrix using orthogonal basis extracted from previous A.

    From the paper: "We initialize A using orthogonal basis extraction via QR
    decomposition from the previous task's fine-tuned A."

    The formula:
        Q, R = QR(A_prev^T)
        Q = Q * sign(diag(R))^T
        A_new_init = Q^T

    This ensures: A_new_init @ A_new_init^T = I_r (orthonormal structure)

    Args:
        A_prev: Previous A matrix (r x d) where r is LoRA rank

    Returns:
        Orthogonally initialized A matrix for new task

    Raises:
        SLAOMergeError: If QR decomposition fails (e.g., singular matrix)
    """
    import torch

    try:
        # QR decomposition of A_prev^T
        Q, R = torch.linalg.qr(A_prev.T)

        # Sign correction to ensure consistent orientation
        # This makes the decomposition unique
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)

        # Return Q^T as the new initialization
        # This has the property that A_init @ A_init^T = I_r
        return Q.T
    except RuntimeError as e:
        raise SLAOMergeError(
            f"Orthogonal initialization failed - QR decomposition error: {e}",
            suggestion="The A matrix may be singular or ill-conditioned"
        ) from e


def merge_B_matrices(
    B_merged: "torch.Tensor",
    B_new: "torch.Tensor",
    scale: float
) -> "torch.Tensor":
    """
    Merge B matrix using time-aware scaling.

    From the paper: "B_merge^i = B_merge^(i-1) + lambda(i) * (B_ft,i - B_merge^(i-1))"

    This is equivalent to exponential moving average with decaying weight.

    Args:
        B_merged: Previously merged B matrix
        B_new: Newly fine-tuned B matrix
        scale: Time-aware scaling factor lambda(i)

    Returns:
        Merged B matrix
    """
    return B_merged + scale * (B_new - B_merged)


def merge_A_matrices(A_new: "torch.Tensor") -> "torch.Tensor":
    """
    Merge A matrix using direct replacement.

    From the paper: "Due to the intrinsic asymmetry of B and A in LoRA,
    we update A_merge^i = A_ft,i"

    A is directly replaced because it captures the input projection which
    benefits from fresh task-specific adaptation.

    Args:
        A_new: Newly fine-tuned A matrix

    Returns:
        New A matrix (direct replacement)
    """
    return A_new.clone()


# =============================================================================
# PHASE 4: ADVANCED SLAO FEATURES
# =============================================================================

def compute_task_similarity(
    lora_state_1: Dict[str, "torch.Tensor"],
    lora_state_2: Dict[str, "torch.Tensor"],
) -> float:
    """
    Compute similarity between two LoRA adapters using cosine similarity.

    Phase 4.1: Task similarity is used to determine how much to preserve
    from the previous run. Similar tasks benefit from more aggressive
    merging, while dissimilar tasks need more preservation.

    Args:
        lora_state_1: First LoRA state dict
        lora_state_2: Second LoRA state dict

    Returns:
        Cosine similarity in range [-1, 1], higher = more similar
    """
    import torch

    # Flatten all B matrices into single vectors for comparison
    # We use B matrices since they capture the output transformation
    vec1_parts = []
    vec2_parts = []

    for key in lora_state_1:
        if ".lora_B." in key and key in lora_state_2:
            v1 = lora_state_1[key]
            v2 = lora_state_2[key]
            if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
                vec1_parts.append(v1.flatten())
                vec2_parts.append(v2.flatten())

    if not vec1_parts:
        # No B matrices found, return neutral similarity
        return 0.0

    vec1 = torch.cat(vec1_parts)
    vec2 = torch.cat(vec2_parts)

    # Compute cosine similarity
    dot_product = torch.sum(vec1 * vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = (dot_product / (norm1 * norm2)).item()
    return similarity


def adaptive_scale(
    base_scale: float,
    similarity: float,
    scale_range: Tuple[float, float] = (0.5, 1.5),
) -> float:
    """
    Adjust scaling factor based on task similarity.

    Phase 4.1: When tasks are similar (high similarity), we can merge more
    aggressively (higher scale). When tasks are dissimilar, we preserve
    more of the previous learning (lower scale).

    Args:
        base_scale: Base scaling factor from time_aware_scale()
        similarity: Task similarity in range [-1, 1]
        scale_range: Multiplier range for adjustment (min, max)

    Returns:
        Adjusted scaling factor
    """
    min_mult, max_mult = scale_range

    # Map similarity [-1, 1] to multiplier [min_mult, max_mult]
    # similarity = 1 (identical) -> max_mult (merge more)
    # similarity = 0 (orthogonal) -> 1.0 (no change)
    # similarity = -1 (opposite) -> min_mult (merge less)
    normalized = (similarity + 1) / 2  # Map to [0, 1]
    multiplier = min_mult + normalized * (max_mult - min_mult)

    return base_scale * multiplier


def get_layer_scale(
    layer_name: str,
    total_layers: int,
    early_scale: float = 0.3,
    middle_scale: float = 0.5,
    late_scale: float = 0.7,
) -> float:
    """
    Get layer-specific scaling factor for selective layer merging.

    Phase 4.2: Different layers capture different features:
    - Early layers: General features, can be merged more aggressively
    - Middle layers: Intermediate representations
    - Late layers: Task-specific features, preserve more

    Args:
        layer_name: Name of the layer (e.g., "model.layers.15.self_attn.q_proj")
        total_layers: Total number of layers in the model
        early_scale: Scale for early layers (0-33%)
        middle_scale: Scale for middle layers (33-66%)
        late_scale: Scale for late layers (66-100%)

    Returns:
        Scale factor for this layer
    """
    import re

    # Extract layer number from name
    # Common patterns: "layers.15.", "h.15.", "block.15."
    match = re.search(r'(?:layers|h|block)\.(\d+)\.', layer_name)

    if not match:
        # Can't determine layer, use middle scale
        return middle_scale

    layer_idx = int(match.group(1))
    layer_position = layer_idx / max(total_layers - 1, 1)  # Normalize to [0, 1]

    if layer_position < 0.33:
        return early_scale
    elif layer_position < 0.66:
        return middle_scale
    else:
        return late_scale


def estimate_total_layers(lora_state: Dict[str, "torch.Tensor"]) -> int:
    """
    Estimate total layers from LoRA state dict.

    Args:
        lora_state: LoRA state dict

    Returns:
        Estimated number of layers
    """
    import re

    max_layer = 0
    for key in lora_state:
        match = re.search(r'(?:layers|h|block)\.(\d+)\.', key)
        if match:
            max_layer = max(max_layer, int(match.group(1)))

    return max_layer + 1  # 0-indexed


# =============================================================================
# SLAO MERGER CLASS
# =============================================================================

class SLAOMerger:
    """
    SLAO (Single LoRA via Asymmetric Merging) merger for continual learning.

    Maintains a single merged LoRA across multiple training runs while
    minimizing catastrophic forgetting through:
    - Orthogonal initialization of A matrices
    - Time-aware scaling for B matrix merging
    - Asymmetric treatment of A (replace) vs B (merge)

    Usage:
        merger = SLAOMerger()

        # Run 1: Train initial LoRA
        lora_1 = train_lora(model, data_chunk_1)
        merger.initialize(lora_1)

        # Run 2+: Train and merge
        lora_2 = train_lora(model, data_chunk_2, init_from=merger.get_init_weights())
        merger.merge(lora_2, run_index=2)

        # Get final merged LoRA
        final_lora = merger.get_merged_lora()
    """

    def __init__(self, config: Optional[SLAOConfig] = None):
        """
        Initialize the SLAO merger.

        Args:
            config: Optional SLAOConfig, uses defaults if not provided
        """
        self.config = config or SLAOConfig()
        self._merged_state: Optional[Dict[str, Any]] = None
        self._run_index: int = 0
        self._merge_history: List[MergeResult] = []

        logger.info(f"SLAOMerger initialized with config: scaling={self.config.scaling_type}")

    def initialize(self, lora_state_dict: Dict[str, "torch.Tensor"]) -> None:
        """
        Initialize the merger with the first LoRA.

        Args:
            lora_state_dict: State dict from first trained LoRA
        """
        import torch

        # Deep copy to avoid modifying original
        self._merged_state = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in lora_state_dict.items()
        }
        self._run_index = 1

        logger.info(f"SLAO initialized with {len(lora_state_dict)} parameters")

    def get_init_weights(self) -> Optional[Dict[str, "torch.Tensor"]]:
        """
        Get initialization weights for the next training run.

        For SLAO:
        - A matrices: Orthogonally initialized from previous A
        - B matrices: Initialized from previous fine-tuned B

        Returns:
            State dict for initializing next LoRA, or None if not initialized
        """
        import torch

        if self._merged_state is None:
            return None

        init_state = {}

        for key, value in self._merged_state.items():
            if not isinstance(value, torch.Tensor):
                init_state[key] = value
                continue

            if ".lora_A." in key and self.config.use_orthogonal_init:
                # Orthogonal initialization for A
                init_state[key] = orthogonal_init_A(value)
                logger.debug(f"Orthogonal init for {key}")
            else:
                # Direct copy for B and other parameters
                init_state[key] = value.clone()

        return init_state

    def merge(
        self,
        new_lora_state: Dict[str, "torch.Tensor"],
        run_index: Optional[int] = None,
    ) -> MergeResult:
        """
        Merge a newly trained LoRA into the accumulated merged LoRA.

        Args:
            new_lora_state: State dict from newly trained LoRA
            run_index: Optional run index (auto-increments if not provided)

        Returns:
            MergeResult with merge statistics
        """
        import torch
        import time

        if self._merged_state is None:
            # First run - initialize
            self.initialize(new_lora_state)
            return MergeResult(
                run_index=1,
                scale_factor=1.0,
                a_matrices_merged=0,
                b_matrices_merged=0,
                total_params_merged=0,
                merge_time_seconds=0.0,
            )

        start_time = time.time()

        # Update run index
        if run_index is not None:
            self._run_index = run_index
        else:
            self._run_index += 1

        # Compute base scaling factor
        base_scale = time_aware_scale(
            self._run_index,
            scaling_type=self.config.scaling_type,
            min_scale=self.config.min_scale,
        )

        # Phase 4.1: Adaptive scaling based on task similarity
        if self.config.use_adaptive_scaling:
            similarity = compute_task_similarity(self._merged_state, new_lora_state)
            scale = adaptive_scale(
                base_scale,
                similarity,
                scale_range=self.config.adaptive_scale_range,
            )
            logger.debug(f"Adaptive scaling: similarity={similarity:.4f}, scale={scale:.4f}")
        else:
            scale = base_scale

        # Phase 4.2: Estimate total layers for layer-specific scaling
        total_layers = None
        if self.config.use_layer_scaling:
            total_layers = estimate_total_layers(new_lora_state)
            logger.debug(f"Layer scaling enabled: {total_layers} layers detected")

        a_count = 0
        b_count = 0
        total_params = 0

        # Merge each parameter
        for key, new_value in new_lora_state.items():
            if not isinstance(new_value, torch.Tensor):
                continue

            if key not in self._merged_state:
                # New parameter - just copy
                self._merged_state[key] = new_value.clone()
                continue

            merged_value = self._merged_state[key]

            # Phase 4.2: Get layer-specific scale if enabled
            if self.config.use_layer_scaling and total_layers:
                layer_scale = get_layer_scale(
                    key,
                    total_layers,
                    early_scale=self.config.layer_scale_early,
                    middle_scale=self.config.layer_scale_middle,
                    late_scale=self.config.layer_scale_late,
                )
                effective_scale = scale * layer_scale
            else:
                effective_scale = scale

            if ".lora_A." in key:
                # A matrix: direct replacement
                self._merged_state[key] = merge_A_matrices(new_value)
                a_count += 1
            elif ".lora_B." in key:
                # B matrix: time-aware merge with layer-specific scale
                self._merged_state[key] = merge_B_matrices(
                    merged_value, new_value, effective_scale
                )
                b_count += 1
            else:
                # Other parameters (e.g., scaling): use weighted average
                self._merged_state[key] = merge_B_matrices(
                    merged_value, new_value, effective_scale
                )

            total_params += new_value.numel()

        merge_time = time.time() - start_time

        result = MergeResult(
            run_index=self._run_index,
            scale_factor=scale,
            a_matrices_merged=a_count,
            b_matrices_merged=b_count,
            total_params_merged=total_params,
            merge_time_seconds=merge_time,
        )

        if self.config.save_merge_history:
            self._merge_history.append(result)

        logger.info(
            f"SLAO merge complete: run={self._run_index}, scale={scale:.4f}, "
            f"A={a_count}, B={b_count}, time={merge_time:.3f}s"
        )

        return result

    def get_merged_lora(self) -> Optional[Dict[str, "torch.Tensor"]]:
        """Get the current merged LoRA state dict."""
        return self._merged_state

    def save(self, path: str) -> None:
        """
        Save the merged LoRA and merge history.

        Args:
            path: Directory path to save to

        Raises:
            SLAOCheckpointError: If save fails
        """
        import torch

        save_dir = Path(path)

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise SLAOCheckpointError(
                "save", str(save_dir),
                f"Permission denied creating directory: {e}"
            ) from e
        except OSError as e:
            raise SLAOCheckpointError(
                "save", str(save_dir),
                f"Failed to create directory: {e}"
            ) from e

        # Save merged weights
        if self._merged_state is not None:
            try:
                torch.save(self._merged_state, save_dir / "merged_lora.pt")
            except Exception as e:
                raise SLAOCheckpointError(
                    "save", str(save_dir / "merged_lora.pt"),
                    f"Failed to save weights: {e}"
                ) from e

        # Save merge history
        history_data = {
            "run_index": self._run_index,
            "config": {
                "scaling_type": self.config.scaling_type,
                "min_scale": self.config.min_scale,
                "use_orthogonal_init": self.config.use_orthogonal_init,
            },
            "history": [
                {
                    "run_index": r.run_index,
                    "scale_factor": r.scale_factor,
                    "a_matrices_merged": r.a_matrices_merged,
                    "b_matrices_merged": r.b_matrices_merged,
                    "total_params_merged": r.total_params_merged,
                    "merge_time_seconds": r.merge_time_seconds,
                }
                for r in self._merge_history
            ]
        }

        try:
            with open(save_dir / "merge_history.json", "w") as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            raise SLAOCheckpointError(
                "save", str(save_dir / "merge_history.json"),
                f"Failed to save history: {e}"
            ) from e

        logger.info(f"SLAO merger saved to {save_dir}")

    def load(self, path: str) -> None:
        """
        Load a previously saved merged LoRA.

        Args:
            path: Directory path to load from

        Raises:
            SLAOCheckpointError: If load fails or checkpoint not found
        """
        import torch

        load_dir = Path(path)

        if not load_dir.exists():
            raise SLAOCheckpointError(
                "load", str(load_dir),
                f"Checkpoint directory not found"
            )

        # Load merged weights
        weights_path = load_dir / "merged_lora.pt"
        if not weights_path.exists():
            raise SLAOCheckpointError(
                "load", str(weights_path),
                f"No merged_lora.pt found in checkpoint"
            )

        try:
            # Security check for PyTorch version
            check_torch_security()
            self._merged_state = torch.load(weights_path, weights_only=True)
        except Exception as e:
            raise SLAOCheckpointError(
                "load", str(weights_path),
                f"Failed to load weights: {e}"
            ) from e

        # Load history
        history_path = load_dir / "merge_history.json"
        if history_path.exists():
            try:
                with open(history_path) as f:
                    history_data = json.load(f)

                self._run_index = history_data.get("run_index", 0)

                # Restore config
                cfg = history_data.get("config", {})
                self.config.scaling_type = cfg.get("scaling_type", "sqrt")
                self.config.min_scale = cfg.get("min_scale", 0.1)
                self.config.use_orthogonal_init = cfg.get("use_orthogonal_init", True)
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted merge history file, using defaults: {e}")
            except Exception as e:
                logger.warning(f"Failed to load merge history: {e}")

        logger.info(f"SLAO merger loaded from {load_dir}, run_index={self._run_index}")

    @property
    def run_index(self) -> int:
        """Current run index."""
        return self._run_index

    @property
    def merge_history(self) -> List[MergeResult]:
        """List of all merge operations performed."""
        return self._merge_history


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def merge_lora_weights(
    base_lora: Dict[str, "torch.Tensor"],
    new_lora: Dict[str, "torch.Tensor"],
    run_index: int = 2,
    method: str = "slao",
) -> Dict[str, "torch.Tensor"]:
    """
    Convenience function to merge two LoRA state dicts.

    Args:
        base_lora: Base LoRA state dict (already merged)
        new_lora: New LoRA state dict to merge in
        run_index: Current run index for time-aware scaling
        method: Merge method ("slao", "average", "replace")

    Returns:
        Merged LoRA state dict
    """
    import torch

    if method == "slao":
        merger = SLAOMerger()
        merger.initialize(base_lora)
        merger.merge(new_lora, run_index=run_index)
        return merger.get_merged_lora()

    elif method == "average":
        # Simple averaging (no time-aware scaling)
        result = {}
        for key in base_lora:
            if isinstance(base_lora[key], torch.Tensor) and key in new_lora:
                result[key] = (base_lora[key] + new_lora[key]) / 2
            else:
                result[key] = base_lora[key]
        return result

    elif method == "replace":
        # Just use the new one
        return {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in new_lora.items()}

    else:
        valid_methods = ("slao", "average", "replace")
        raise InvalidSettingError(
            "method", method, f"one of {valid_methods}",
            suggestion="Use 'slao' for best continual learning results"
        )
