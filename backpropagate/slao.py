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

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .exceptions import (
    BackpropagateError,
    InvalidSettingError,
    SLAOCheckpointError,
    SLAOMergeError,
)
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
    adaptive_scale_range: tuple[float, float] = (0.5, 1.5)  # Scale multiplier range

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
    a_norm_before: float | None = None
    a_norm_after: float | None = None
    b_norm_before: float | None = None
    b_norm_after: float | None = None


# =============================================================================
# CORE SLAO FUNCTIONS
# =============================================================================

def time_aware_scale(
    run_index: int,
    scaling_type: str | Any = "sqrt",
    min_scale: float = 0.1,
) -> float:
    """
    Compute time-aware scaling factor for SLAO merging.

    From the paper: "lambda(i) = 1/sqrt(i) is a natural choice for the scaling
    factor" because task vectors from different tasks tend to be approximately
    orthogonal.

    Stage C amend BACKEND-B-018: ``scaling_type`` accepts either a string
    (one of the built-in schedules) OR a callable ``(run_index) -> float``
    for operators experimenting with custom decay curves without forking
    the merger. The callable's return value is still clamped to
    ``[min_scale, 1.0]`` so a buggy custom schedule can't produce wildly
    out-of-range scales.

    Args:
        run_index: Current run index (1-based, first run = 1)
        scaling_type: Type of scaling. Accepts:
            - "sqrt": 1/√i (paper default, good balance)
            - "linear": 1/i (more aggressive, preserves early learning)
            - "log": 1/log(i+1) (slower decay, more plasticity)
            - "constant": 1.0 (simple EMA, no decay)
            - ``Callable[[int], float]``: a custom decay schedule. The
              callable receives ``run_index`` and returns a raw scale
              that's clamped to ``[min_scale, 1.0]`` after the call.
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
        >>> time_aware_scale(4, lambda i: 1.0 / (i ** 0.3))  # custom
    """
    if not isinstance(run_index, int) or run_index < 1:
        raise InvalidSettingError(
            "run_index", run_index, "positive integer >= 1",
            suggestion="Run index should start at 1 for the first run"
        )

    # Stage C amend BACKEND-B-018: handle callable schedule first so we
    # bypass the string-validation branch entirely. Failures inside the
    # custom callable propagate to the caller (no silent fallback —
    # the operator's contract is "if your callable raises, the merge
    # raises").
    if callable(scaling_type):
        try:
            scale = float(scaling_type(run_index))
        except Exception as exc:
            raise InvalidSettingError(
                "scaling_type",
                repr(scaling_type),
                "callable returning a finite float",
                suggestion=(
                    f"The custom scaling callable raised "
                    f"{type(exc).__name__}: {exc}. Return a float in "
                    f"[min_scale, 1.0]."
                ),
            ) from exc
        if not math.isfinite(scale):
            raise InvalidSettingError(
                "scaling_type",
                repr(scaling_type),
                "callable returning a finite float",
                suggestion=(
                    f"Custom callable returned {scale!r} at run_index="
                    f"{run_index}. Return a finite float in [min_scale, 1.0]."
                ),
            )
        return max(scale, min_scale)

    valid_scaling_types = ("sqrt", "linear", "log", "constant")
    if scaling_type not in valid_scaling_types:
        raise InvalidSettingError(
            "scaling_type", scaling_type, f"one of {valid_scaling_types} (or a callable)",
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
    else:  # constant — validated above
        # No decay (simple averaging)
        scale = 1.0

    return max(scale, min_scale)


def orthogonal_init_A(A_prev: torch.Tensor) -> torch.Tensor:
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
        result: torch.Tensor = Q.T
        return result
    except RuntimeError as e:
        raise SLAOMergeError(
            f"Orthogonal initialization failed - QR decomposition error: {e}",
            suggestion="The A matrix may be singular or ill-conditioned"
        ) from e


def merge_B_matrices(
    B_merged: torch.Tensor,
    B_new: torch.Tensor,
    scale: float
) -> torch.Tensor:
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


def merge_A_matrices(A_new: torch.Tensor) -> torch.Tensor:
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
    lora_state_1: dict[str, torch.Tensor],
    lora_state_2: dict[str, torch.Tensor],
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

    similarity: float = (dot_product / (norm1 * norm2)).item()
    return similarity


def adaptive_scale(
    base_scale: float,
    similarity: float,
    scale_range: tuple[float, float] = (0.5, 1.5),
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


def estimate_total_layers(lora_state: dict[str, torch.Tensor]) -> int:
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

    def __init__(self, config: SLAOConfig | None = None):
        """
        Initialize the SLAO merger.

        Args:
            config: Optional SLAOConfig, uses defaults if not provided
        """
        self.config = config or SLAOConfig()
        self._merged_state: dict[str, Any] | None = None
        self._run_index: int = 0
        self._merge_history: list[MergeResult] = []

        logger.info(f"SLAOMerger initialized with config: scaling={self.config.scaling_type}")

    def initialize(self, lora_state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the merger with the first LoRA.

        Stage C amend BACKEND-B-012: log a/b matrix counts (and DEBUG-level
        layer indices) at initialization so post-mortem triage on PEFT
        version drift or target_modules misconfig can confirm the merger
        saw the expected adapter geometry without re-running the session.
        Pre-fix, only the total parameter count was logged — useful as a
        sanity check but invisible to "did I get all the layers I asked
        for?" questions.

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

        # Stage C amend BACKEND-B-012: a-matrix / b-matrix counts +
        # layer-index breakdown for observability.
        a_keys = [k for k in lora_state_dict if ".lora_A." in k]
        b_keys = [k for k in lora_state_dict if ".lora_B." in k]
        other_keys = [
            k for k in lora_state_dict
            if ".lora_A." not in k and ".lora_B." not in k
        ]
        logger.info(
            f"SLAO initialized with {len(lora_state_dict)} parameters "
            f"(a_matrices={len(a_keys)} b_matrices={len(b_keys)} "
            f"other={len(other_keys)})"
        )
        if a_keys or b_keys:
            # DEBUG-level breakdown — only fires when logging is verbose.
            # Extract numeric layer index from common PEFT key patterns
            # like ``base_model.model.model.layers.<N>.self_attn.q_proj.lora_A.default.weight``.
            import re
            _LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
            a_layers = sorted({
                int(m.group(1))
                for k in a_keys
                for m in [_LAYER_RE.search(k)]
                if m
            })
            b_layers = sorted({
                int(m.group(1))
                for k in b_keys
                for m in [_LAYER_RE.search(k)]
                if m
            })
            if a_layers:
                logger.debug(
                    f"SLAO initialize: a_matrix layer indices = "
                    f"{a_layers!r} (first={a_layers[0]} last={a_layers[-1]} "
                    f"count={len(a_layers)})"
                )
            if b_layers:
                logger.debug(
                    f"SLAO initialize: b_matrix layer indices = "
                    f"{b_layers!r} (first={b_layers[0]} last={b_layers[-1]} "
                    f"count={len(b_layers)})"
                )

    def get_init_weights(self) -> dict[str, torch.Tensor] | None:
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
        new_lora_state: dict[str, torch.Tensor],
        run_index: int | None = None,
        run_id: str | None = None,
    ) -> MergeResult:
        """
        Merge a newly trained LoRA into the accumulated merged LoRA.

        Args:
            new_lora_state: State dict from newly trained LoRA
            run_index: Optional run index (auto-increments if not provided)
            run_id: Optional correlation token threaded into log lines and the
                error envelope if a divergence (NaN/inf) is detected — see B-001
                / B-008. Used for triage, never for math.

        Returns:
            MergeResult with merge statistics. Populates ``a_norm_before /
            after`` and ``b_norm_before / after`` (sampled from one
            representative A and B matrix) so post-mortems can detect silent
            corruption without re-running the merge.

        Raises:
            BackpropagateError(code="SLAO_MERGE_DIVERGED"): If a representative
                merged A or B matrix contains NaN or inf after the merge step.
                The error carries the run_index, run_id, and offending layer
                so an operator can rewind to the last healthy checkpoint.
        """
        import time

        import torch

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

        # B-008: capture sampled before-norms on one representative A and B
        # matrix so we can populate MergeResult.{a,b}_norm_{before,after} for
        # post-mortem observability. We pick the FIRST matched key per matrix
        # type (deterministic given dict ordering on Python 3.7+).
        rep_a_key: str | None = None
        rep_b_key: str | None = None
        a_norm_before: float | None = None
        b_norm_before: float | None = None
        for key in new_lora_state:
            if rep_a_key is None and ".lora_A." in key and key in self._merged_state:
                rep_a_key = key
                tensor = self._merged_state[key]
                if isinstance(tensor, torch.Tensor):
                    a_norm_before = float(tensor.detach().float().norm().item())
            if rep_b_key is None and ".lora_B." in key and key in self._merged_state:
                rep_b_key = key
                tensor = self._merged_state[key]
                if isinstance(tensor, torch.Tensor):
                    b_norm_before = float(tensor.detach().float().norm().item())
            if rep_a_key is not None and rep_b_key is not None:
                break

        # Stage C BACKEND-B-008: full-scan divergence detection. The Stage A
        # implementation only checked the FIRST matched A and B keys (the
        # representative sample). A NaN that lands in layer 27 of a 32-layer
        # model would pass the check and silently propagate. Folding a
        # ``torch.isfinite(...).all()`` probe into the existing per-key
        # iteration costs ~one extra reduction per parameter (microseconds
        # per layer; ~10ms total for a 7B LoRA) — defense-in-depth without
        # sampling. We capture the FIRST non-finite key so the error envelope
        # carries a precise diagnostic instead of "a NaN appeared somewhere."
        first_bad_key: str | None = None
        first_bad_kind: str | None = None  # "lora_A" / "lora_B" / "other"

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
                kind = "lora_A"
            elif ".lora_B." in key:
                # B matrix: time-aware merge with layer-specific scale
                self._merged_state[key] = merge_B_matrices(
                    merged_value, new_value, effective_scale
                )
                b_count += 1
                kind = "lora_B"
            else:
                # Other parameters (e.g., scaling): use weighted average
                self._merged_state[key] = merge_B_matrices(
                    merged_value, new_value, effective_scale
                )
                kind = "other"

            # Stage C BACKEND-B-008: fold the finite-check into the same loop
            # iteration so we don't pay for a second pass over the state dict.
            # ``torch.isfinite(x).all()`` is a single reduction kernel call.
            if first_bad_key is None:
                merged_after = self._merged_state[key]
                if isinstance(merged_after, torch.Tensor):
                    try:
                        if not torch.isfinite(merged_after).all().item():
                            first_bad_key = key
                            first_bad_kind = kind
                    except RuntimeError as probe_err:
                        # An OOM or dtype mismatch on the probe itself
                        # shouldn't take the merge down; record it and move
                        # on. The error gets logged but the merge succeeds.
                        logger.debug(
                            f"finite-probe failed for {key}: {probe_err}"
                        )

            total_params += new_value.numel()

        # B-008: sample after-norms on the same representative keys so the
        # before/after pair is comparable across the merge step. (Preserved
        # for MergeResult.{a,b}_norm_after observability — the divergence
        # check is now full-scan above; the sampled norms remain for
        # post-mortem trend analysis on a representative layer.)
        a_norm_after: float | None = None
        b_norm_after: float | None = None
        if rep_a_key is not None and rep_a_key in self._merged_state:
            tensor = self._merged_state[rep_a_key]
            if isinstance(tensor, torch.Tensor):
                a_norm_after = float(tensor.detach().float().norm().item())
        if rep_b_key is not None and rep_b_key in self._merged_state:
            tensor = self._merged_state[rep_b_key]
            if isinstance(tensor, torch.Tensor):
                b_norm_after = float(tensor.detach().float().norm().item())

        # Stage C BACKEND-B-008: invariant — EVERY merged parameter must be
        # finite. The previous implementation only checked the representative
        # A/B keys; a non-finite weight in a non-sampled layer would silently
        # propagate to the next run. We now use the full-scan result from the
        # merge loop above. The representative-norm sampling is kept for the
        # MergeResult observability surface (operators inspecting the trend
        # of norms across runs).
        def _is_finite(x: float | None) -> bool:
            if x is None:
                return True  # nothing was sampled — neutral
            return math.isfinite(x)

        # Promote a representative-norm non-finite to the loop-detected one if
        # the loop missed it (e.g. probe RuntimeError raced ahead of detection).
        if first_bad_key is None and (
            not _is_finite(a_norm_after) or not _is_finite(b_norm_after)
        ):
            first_bad_key = rep_a_key if not _is_finite(a_norm_after) else rep_b_key
            first_bad_kind = "lora_A" if not _is_finite(a_norm_after) else "lora_B"

        if first_bad_key is not None:
            logger.warning(
                f"SLAO merge diverged: run_index={self._run_index} "
                f"run_id={run_id} layer={first_bad_key} "
                f"kind={first_bad_kind} "
                f"a_norm_after={a_norm_after} b_norm_after={b_norm_after} "
                f"(full-scan; first non-finite key reported)"
            )
            raise BackpropagateError(
                f"SLAO merge produced non-finite weights at run {self._run_index} "
                f"(layer={first_bad_key})",
                code="SLAO_MERGE_DIVERGED",
                details={
                    "run_index": self._run_index,
                    "run_id": run_id,
                    "layer": first_bad_key,
                    "kind": first_bad_kind,
                    "a_norm_after": a_norm_after,
                    "b_norm_after": b_norm_after,
                    "scan_mode": "full",
                },
                suggestion=(
                    "Rewind to the previous healthy checkpoint and inspect "
                    "the latest training run for bf16 underflow, exploding "
                    "gradients, or a corrupted PEFT adapter."
                ),
                retryable=False,
            )

        merge_time = time.time() - start_time

        result = MergeResult(
            run_index=self._run_index,
            scale_factor=scale,
            a_matrices_merged=a_count,
            b_matrices_merged=b_count,
            total_params_merged=total_params,
            merge_time_seconds=merge_time,
            a_norm_before=a_norm_before,
            a_norm_after=a_norm_after,
            b_norm_before=b_norm_before,
            b_norm_after=b_norm_after,
        )

        if self.config.save_merge_history:
            self._merge_history.append(result)

        logger.info(
            f"SLAO merge complete: run={self._run_index}, scale={scale:.4f}, "
            f"A={a_count}, B={b_count}, time={merge_time:.3f}s"
        )

        return result

    def get_merged_lora(self) -> dict[str, torch.Tensor] | None:
        """Get the current merged LoRA state dict."""
        return self._merged_state

    def save(self, path: str, run_id: str | None = None) -> None:
        """
        Save the merged LoRA and merge history atomically.

        B-006: writes flow into a sibling ``<path>.partial`` directory first,
        then ``shutil.move()`` promotes it to the final path. If anything
        raises mid-write the ``.partial`` directory is removed via the
        ``finally`` clause so the operator never sees a half-written SLAO
        checkpoint (config.json present, weights missing).

        Args:
            path: Final directory path. Existing contents at this path are
                overwritten only on successful promotion.
            run_id: Optional correlation token persisted into merge_history.json
                under ``run_id`` so operators can grep one identifier across
                logs + manifests + SLAO history (see B-001).

        Raises:
            SLAOCheckpointError: If save fails. The partial directory is
                cleaned up before the exception escapes.
        """
        import shutil

        import torch

        save_dir = Path(path)
        partial_dir = save_dir.with_name(save_dir.name + ".partial")

        # Pre-flight: parent directory must be creatable, partial slot must be
        # clean. We deliberately do NOT mkdir(save_dir) — the atomic promote
        # at the end is responsible for placing the directory.
        try:
            save_dir.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise SLAOCheckpointError(
                "save", str(save_dir),
                f"Permission denied creating parent directory: {e}"
            ) from e
        except OSError as e:
            raise SLAOCheckpointError(
                "save", str(save_dir),
                f"Failed to create parent directory: {e}"
            ) from e

        # Wipe any leftover partial from a prior crash before we start.
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)

        try:
            partial_dir.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            raise SLAOCheckpointError(
                "save", str(partial_dir),
                f"Failed to create partial directory: {e}"
            ) from e

        try:
            # Save merged weights
            if self._merged_state is not None:
                try:
                    torch.save(self._merged_state, partial_dir / "merged_lora.pt")
                except Exception as e:
                    raise SLAOCheckpointError(
                        "save", str(partial_dir / "merged_lora.pt"),
                        f"Failed to save weights: {e}"
                    ) from e

            # Save merge history (B-001: include run_id correlation token,
            # B-008: include norms so post-mortems can detect drift over time).
            #
            # Stage C amend BACKEND-B-003 / BACKEND-B-004: persist EVERY
            # SLAOConfig field that affects merge math (adaptive_scaling,
            # layer_scaling, layer_scale_*, use_time_aware_scaling,
            # normalize_after_merge). Pre-fix, load() restored only three
            # fields (scaling_type / min_scale / use_orthogonal_init);
            # resuming a session that started with ``adaptive_scaling=True``
            # silently flipped it back to False on the resumed runs,
            # producing different math than the prior session. Operators
            # could only detect this by diff-ing logs across sessions.
            # The ``version`` field anchors the schema for v1.4 migration.
            history_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "run_index": self._run_index,
                "run_id": run_id,
                "config": {
                    "scaling_type": self.config.scaling_type,
                    "min_scale": self.config.min_scale,
                    "use_orthogonal_init": self.config.use_orthogonal_init,
                    "use_time_aware_scaling": self.config.use_time_aware_scaling,
                    "normalize_after_merge": self.config.normalize_after_merge,
                    "save_merge_history": self.config.save_merge_history,
                    # Phase 4.1 / 4.2 fields — silently dropped pre-fix.
                    "use_adaptive_scaling": self.config.use_adaptive_scaling,
                    "adaptive_scale_range": list(self.config.adaptive_scale_range),
                    "use_layer_scaling": self.config.use_layer_scaling,
                    "layer_scale_early": self.config.layer_scale_early,
                    "layer_scale_middle": self.config.layer_scale_middle,
                    "layer_scale_late": self.config.layer_scale_late,
                },
                "history": [
                    {
                        "run_index": r.run_index,
                        "scale_factor": r.scale_factor,
                        "a_matrices_merged": r.a_matrices_merged,
                        "b_matrices_merged": r.b_matrices_merged,
                        "total_params_merged": r.total_params_merged,
                        "merge_time_seconds": r.merge_time_seconds,
                        "a_norm_before": r.a_norm_before,
                        "a_norm_after": r.a_norm_after,
                        "b_norm_before": r.b_norm_before,
                        "b_norm_after": r.b_norm_after,
                    }
                    for r in self._merge_history
                ]
            }

            try:
                with open(partial_dir / "merge_history.json", "w") as f:
                    json.dump(history_data, f, indent=2)
            except Exception as e:
                raise SLAOCheckpointError(
                    "save", str(partial_dir / "merge_history.json"),
                    f"Failed to save history: {e}"
                ) from e

            # Atomic promote: if save_dir already exists (re-save), remove it
            # first so shutil.move drops the partial into place cleanly. The
            # window between rmtree(save_dir) and move(partial) is tiny but
            # NOT atomic across processes — single-process MultiRunTrainer
            # callers (the only callers in this repo) are unaffected.
            if save_dir.exists():
                shutil.rmtree(save_dir)
            shutil.move(str(partial_dir), str(save_dir))
        except SLAOCheckpointError:
            raise
        except Exception as e:
            raise SLAOCheckpointError(
                "save", str(save_dir),
                f"Atomic promotion failed: {e}"
            ) from e
        finally:
            # Belt-and-braces: if partial_dir still exists after any failure
            # path, remove it so a retry has a clean slot.
            if partial_dir.exists():
                shutil.rmtree(partial_dir, ignore_errors=True)

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
                "Checkpoint directory not found"
            )

        # Load merged weights
        weights_path = load_dir / "merged_lora.pt"
        if not weights_path.exists():
            raise SLAOCheckpointError(
                "load", str(weights_path),
                "No merged_lora.pt found in checkpoint"
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

                # Stage C amend BACKEND-B-003: read+verify schema version. A
                # mismatched version is non-fatal at load time (we use field
                # defaults for missing fields) but the operator gets a clear
                # WARN line so they can correlate post-resume divergence with
                # a schema mismatch. v1.4 will add a real migrator; v1.3
                # just fails-loud-but-keeps-going.
                disk_version = str(history_data.get("version") or "0.0")
                _CURRENT_SLAO_SCHEMA = "1.0"
                if disk_version != _CURRENT_SLAO_SCHEMA:
                    logger.warning(
                        f"SLAO merge_history.json on disk has "
                        f"version={disk_version!r} but this build expects "
                        f"{_CURRENT_SLAO_SCHEMA!r}. Missing fields will fall "
                        f"back to runtime defaults — pass them explicitly on "
                        f"the resumed session to silence this warning and "
                        f"preserve prior merge math."
                    )

                self._run_index = history_data.get("run_index", 0)

                # Restore config.
                #
                # Stage C amend BACKEND-B-004: restore EVERY field the save
                # path writes. Pre-fix, ``load`` quietly dropped
                # ``use_adaptive_scaling`` / ``use_layer_scaling`` /
                # ``layer_scale_*`` so a resumed session silently changed
                # merge math. The fallback-to-default branches each emit a
                # warn-once line so the operator knows which fields fell
                # back and why — the message is actionable ("pass the flag
                # explicitly to silence this warning").
                cfg = history_data.get("config", {})
                self.config.scaling_type = cfg.get("scaling_type", "sqrt")
                self.config.min_scale = cfg.get("min_scale", 0.1)
                self.config.use_orthogonal_init = cfg.get("use_orthogonal_init", True)

                # Optional fields that LANDED in v1.3 schema. Pre-v1.3
                # checkpoints don't carry them; restore from runtime
                # default + WARN once per missing field so the operator
                # sees the fall-back surface.
                def _restore_or_warn(field_name: str, default: Any) -> Any:
                    if field_name in cfg:
                        return cfg[field_name]
                    logger.warning(
                        f"Resuming from a pre-v1.3 SLAO checkpoint that "
                        f"does not carry {field_name!r}; defaulting to "
                        f"{default!r}. Pass {field_name} explicitly on the "
                        f"resumed session to silence this warning and "
                        f"preserve prior merge math."
                    )
                    return default

                self.config.use_time_aware_scaling = _restore_or_warn(
                    "use_time_aware_scaling", self.config.use_time_aware_scaling
                )
                self.config.normalize_after_merge = _restore_or_warn(
                    "normalize_after_merge", self.config.normalize_after_merge
                )
                self.config.use_adaptive_scaling = _restore_or_warn(
                    "use_adaptive_scaling", self.config.use_adaptive_scaling
                )
                if "adaptive_scale_range" in cfg:
                    rng = cfg["adaptive_scale_range"]
                    if isinstance(rng, (list, tuple)) and len(rng) == 2:
                        self.config.adaptive_scale_range = (float(rng[0]), float(rng[1]))
                self.config.use_layer_scaling = _restore_or_warn(
                    "use_layer_scaling", self.config.use_layer_scaling
                )
                self.config.layer_scale_early = _restore_or_warn(
                    "layer_scale_early", self.config.layer_scale_early
                )
                self.config.layer_scale_middle = _restore_or_warn(
                    "layer_scale_middle", self.config.layer_scale_middle
                )
                self.config.layer_scale_late = _restore_or_warn(
                    "layer_scale_late", self.config.layer_scale_late
                )
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
    def merge_history(self) -> list[MergeResult]:
        """List of all merge operations performed."""
        return self._merge_history


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def merge_lora_weights(
    base_lora: dict[str, torch.Tensor],
    new_lora: dict[str, torch.Tensor],
    run_index: int = 2,
    method: str = "slao",
) -> dict[str, torch.Tensor]:
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
        merged = merger.get_merged_lora()
        assert merged is not None  # guaranteed after initialize()
        return merged

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
