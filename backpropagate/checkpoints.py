"""
Checkpoint Management for Multi-Run Training
=============================================

Phase 5.3: Smart checkpoint pruning to manage disk space while preserving
the most valuable checkpoints.

Features:
- Keep best N checkpoints by validation loss
- Always preserve final checkpoint
- Optionally keep run boundary checkpoints
- Automatic cleanup after each run
- Manifest tracking with metadata

Usage:
    from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy

    policy = CheckpointPolicy(keep_best_n=3, keep_final=True)
    manager = CheckpointManager(checkpoint_dir, policy)

    # After each run
    manager.register(run_idx, checkpoint_path, val_loss=0.5)
    manager.prune()  # Automatically removes low-value checkpoints

    # Get stats
    print(manager.get_stats())  # Total size, count, best checkpoint
"""

import json
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointPolicy",
    "CheckpointInfo",
    "CheckpointStats",
    "CheckpointManager",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CheckpointPolicy:
    """
    Policy for checkpoint retention and pruning.

    Attributes:
        keep_best_n: Keep the N checkpoints with lowest validation loss
        keep_final: Always keep the most recent checkpoint
        keep_run_boundaries: Keep the first checkpoint of each run
        max_total: Hard limit on total checkpoints (0 = unlimited)
        min_improvement: Only keep if loss improved by at least this much
        auto_prune: Automatically prune after each registration
    """
    keep_best_n: int = 3
    keep_final: bool = True
    keep_run_boundaries: bool = False
    max_total: int = 10
    min_improvement: float = 0.0
    auto_prune: bool = True


@dataclass
class CheckpointInfo:
    """
    Metadata for a single checkpoint.

    Attributes:
        run_index: Which run this checkpoint is from
        path: Path to the checkpoint directory/file
        validation_loss: Validation loss at this checkpoint (lower = better)
        training_loss: Training loss at this checkpoint
        timestamp: When the checkpoint was created
        is_run_boundary: True if this is the first checkpoint of a run
        is_final: True if this is the most recent checkpoint
        size_bytes: Size of the checkpoint in bytes
        protected: If True, this checkpoint won't be pruned
    """
    run_index: int
    path: str
    validation_loss: Optional[float] = None
    training_loss: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_run_boundary: bool = False
    is_final: bool = False
    size_bytes: int = 0
    protected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        return cls(**data)


@dataclass
class CheckpointStats:
    """
    Statistics about managed checkpoints.

    Attributes:
        total_count: Number of checkpoints
        total_size_bytes: Total size in bytes
        total_size_gb: Total size in gigabytes
        best_checkpoint: Info about the best checkpoint (lowest val loss)
        oldest_checkpoint: Info about the oldest checkpoint
        newest_checkpoint: Info about the newest checkpoint
        protected_count: Number of protected checkpoints
        prunable_count: Number of checkpoints that can be pruned
    """
    total_count: int = 0
    total_size_bytes: int = 0
    total_size_gb: float = 0.0
    best_checkpoint: Optional[CheckpointInfo] = None
    oldest_checkpoint: Optional[CheckpointInfo] = None
    newest_checkpoint: Optional[CheckpointInfo] = None
    protected_count: int = 0
    prunable_count: int = 0

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Checkpoints: {self.total_count} ({self.total_size_gb:.2f} GB)",
            f"Protected: {self.protected_count}, Prunable: {self.prunable_count}",
        ]
        if self.best_checkpoint and self.best_checkpoint.validation_loss is not None:
            lines.append(f"Best: Run {self.best_checkpoint.run_index} (val_loss={self.best_checkpoint.validation_loss:.4f})")
        return "\n".join(lines)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Manages checkpoints with smart pruning based on validation loss.

    The manager maintains a manifest file (manifest.json) in the checkpoint
    directory that tracks all checkpoints and their metadata.
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        checkpoint_dir: str,
        policy: Optional[CheckpointPolicy] = None,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored
            policy: Retention policy (uses defaults if not provided)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.policy = policy or CheckpointPolicy()
        self._checkpoints: List[CheckpointInfo] = []
        self._manifest_path = self.checkpoint_dir / self.MANIFEST_FILE

        # Create directory if needed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load checkpoint manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r") as f:
                    data = json.load(f)
                self._checkpoints = [
                    CheckpointInfo.from_dict(c) for c in data.get("checkpoints", [])
                ]
                logger.debug(f"Loaded {len(self._checkpoints)} checkpoints from manifest")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
                self._checkpoints = []
        else:
            self._checkpoints = []

    def _save_manifest(self) -> None:
        """Save checkpoint manifest to disk."""
        try:
            data = {
                "version": "1.0",
                "updated": datetime.now().isoformat(),
                "policy": asdict(self.policy),
                "checkpoints": [c.to_dict() for c in self._checkpoints],
            }
            with open(self._manifest_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved manifest with {len(self._checkpoints)} checkpoints")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def _get_checkpoint_size(self, path: str) -> int:
        """Get total size of a checkpoint directory/file in bytes."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return 0

        if checkpoint_path.is_file():
            return checkpoint_path.stat().st_size

        # Directory - sum all files
        total = 0
        for f in checkpoint_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def register(
        self,
        run_index: int,
        checkpoint_path: str,
        validation_loss: Optional[float] = None,
        training_loss: Optional[float] = None,
        is_run_boundary: bool = False,
        protected: bool = False,
    ) -> CheckpointInfo:
        """
        Register a new checkpoint.

        Args:
            run_index: Which run this checkpoint is from
            checkpoint_path: Path to the checkpoint
            validation_loss: Validation loss (used for ranking)
            training_loss: Training loss
            is_run_boundary: True if this is the start of a new run
            protected: If True, this checkpoint won't be pruned

        Returns:
            CheckpointInfo for the registered checkpoint
        """
        # Mark all existing checkpoints as not final
        for cp in self._checkpoints:
            cp.is_final = False

        # Create new checkpoint info
        size = self._get_checkpoint_size(checkpoint_path)
        info = CheckpointInfo(
            run_index=run_index,
            path=checkpoint_path,
            validation_loss=validation_loss,
            training_loss=training_loss,
            is_run_boundary=is_run_boundary,
            is_final=True,  # This is now the latest
            size_bytes=size,
            protected=protected,
        )

        self._checkpoints.append(info)
        self._save_manifest()

        val_str = f"{validation_loss:.4f}" if validation_loss is not None else "N/A"
        logger.info(
            f"Registered checkpoint: run={run_index}, "
            f"val_loss={val_str}, "
            f"size={size / (1024**2):.1f} MB"
        )

        # Auto-prune if enabled
        if self.policy.auto_prune:
            self.prune()

        return info

    def _score_checkpoint(self, cp: CheckpointInfo) -> float:
        """
        Score a checkpoint for retention (higher = more likely to keep).

        Returns:
            Score value (higher = more valuable)
        """
        score = 0.0

        # Protected checkpoints get infinite score
        if cp.protected:
            return float('inf')

        # Final checkpoint bonus
        if cp.is_final and self.policy.keep_final:
            score += 1000.0

        # Run boundary bonus
        if cp.is_run_boundary and self.policy.keep_run_boundaries:
            score += 500.0

        # Validation loss scoring (lower loss = higher score)
        if cp.validation_loss is not None:
            # Rank by validation loss - best gets highest bonus
            all_losses = [c.validation_loss for c in self._checkpoints if c.validation_loss is not None]
            if all_losses:
                sorted_losses = sorted(all_losses)
                try:
                    rank = sorted_losses.index(cp.validation_loss)
                    # Top N get bonus points
                    if rank < self.policy.keep_best_n:
                        score += 100.0 * (self.policy.keep_best_n - rank)
                except ValueError:
                    pass

        return score

    def _get_prunable_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of checkpoints that can be pruned."""
        prunable = []
        for cp in self._checkpoints:
            if cp.protected:
                continue
            if cp.is_final and self.policy.keep_final:
                continue
            if cp.is_run_boundary and self.policy.keep_run_boundaries:
                continue

            # Check if in top N by validation loss
            if cp.validation_loss is not None:
                all_losses = sorted([
                    c.validation_loss for c in self._checkpoints
                    if c.validation_loss is not None
                ])
                try:
                    rank = all_losses.index(cp.validation_loss)
                    if rank < self.policy.keep_best_n:
                        continue  # In top N, don't prune
                except ValueError:
                    pass

            prunable.append(cp)

        return prunable

    def prune(self, dry_run: bool = False) -> List[CheckpointInfo]:
        """
        Prune checkpoints according to policy.

        Args:
            dry_run: If True, don't actually delete, just return what would be pruned

        Returns:
            List of checkpoints that were (or would be) pruned
        """
        # Score all checkpoints
        scored = [(self._score_checkpoint(cp), cp) for cp in self._checkpoints]
        scored.sort(key=lambda x: x[0], reverse=True)  # Highest scores first

        # Determine which to keep
        to_keep = []
        to_prune = []

        for score, cp in scored:
            # Always keep protected
            if cp.protected:
                to_keep.append(cp)
                continue

            # Check max_total limit
            if self.policy.max_total > 0 and len(to_keep) >= self.policy.max_total:
                to_prune.append(cp)
                continue

            # Keep if score is positive (has some value)
            if score > 0:
                to_keep.append(cp)
            else:
                to_prune.append(cp)

        if not to_prune:
            logger.debug("No checkpoints to prune")
            return []

        if dry_run:
            logger.info(f"Dry run: would prune {len(to_prune)} checkpoints")
            return to_prune

        # Actually delete
        pruned = []
        freed_bytes = 0

        for cp in to_prune:
            try:
                checkpoint_path = Path(cp.path)
                if checkpoint_path.exists():
                    if checkpoint_path.is_dir():
                        shutil.rmtree(checkpoint_path)
                    else:
                        checkpoint_path.unlink()
                    freed_bytes += cp.size_bytes
                    logger.info(f"Pruned checkpoint: run={cp.run_index}, freed={cp.size_bytes / (1024**2):.1f} MB")

                pruned.append(cp)
                self._checkpoints.remove(cp)

            except Exception as e:
                logger.error(f"Failed to prune checkpoint {cp.path}: {e}")

        self._save_manifest()

        logger.info(
            f"Pruned {len(pruned)} checkpoints, "
            f"freed {freed_bytes / (1024**3):.2f} GB"
        )

        return pruned

    def get_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the checkpoint with lowest validation loss."""
        valid = [cp for cp in self._checkpoints if cp.validation_loss is not None]
        if not valid:
            return None
        return min(valid, key=lambda x: x.validation_loss)

    def get_final_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint."""
        final = [cp for cp in self._checkpoints if cp.is_final]
        return final[0] if final else None

    def get_stats(self) -> CheckpointStats:
        """Get statistics about managed checkpoints."""
        if not self._checkpoints:
            return CheckpointStats()

        total_size = sum(cp.size_bytes for cp in self._checkpoints)
        prunable = self._get_prunable_checkpoints()
        protected = [cp for cp in self._checkpoints if cp.protected]

        # Sort by timestamp to find oldest/newest
        sorted_by_time = sorted(self._checkpoints, key=lambda x: x.timestamp)

        return CheckpointStats(
            total_count=len(self._checkpoints),
            total_size_bytes=total_size,
            total_size_gb=total_size / (1024**3),
            best_checkpoint=self.get_best_checkpoint(),
            oldest_checkpoint=sorted_by_time[0] if sorted_by_time else None,
            newest_checkpoint=sorted_by_time[-1] if sorted_by_time else None,
            protected_count=len(protected),
            prunable_count=len(prunable),
        )

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of all registered checkpoints."""
        return list(self._checkpoints)

    def protect_checkpoint(self, run_index: int) -> bool:
        """
        Protect a checkpoint from pruning.

        Args:
            run_index: Run index of checkpoint to protect

        Returns:
            True if checkpoint was found and protected
        """
        for cp in self._checkpoints:
            if cp.run_index == run_index:
                cp.protected = True
                self._save_manifest()
                logger.info(f"Protected checkpoint: run={run_index}")
                return True
        return False

    def unprotect_checkpoint(self, run_index: int) -> bool:
        """
        Remove protection from a checkpoint.

        Args:
            run_index: Run index of checkpoint to unprotect

        Returns:
            True if checkpoint was found and unprotected
        """
        for cp in self._checkpoints:
            if cp.run_index == run_index:
                cp.protected = False
                self._save_manifest()
                logger.info(f"Unprotected checkpoint: run={run_index}")
                return True
        return False

    def cleanup_orphaned(self) -> int:
        """
        Remove checkpoints from manifest that no longer exist on disk.

        Returns:
            Number of orphaned entries removed
        """
        orphaned = []
        for cp in self._checkpoints:
            if not Path(cp.path).exists():
                orphaned.append(cp)

        for cp in orphaned:
            self._checkpoints.remove(cp)
            logger.info(f"Removed orphaned manifest entry: {cp.path}")

        if orphaned:
            self._save_manifest()

        return len(orphaned)

    def force_prune_to_size(self, max_size_gb: float) -> List[CheckpointInfo]:
        """
        Force prune checkpoints until total size is under limit.

        Args:
            max_size_gb: Maximum total size in gigabytes

        Returns:
            List of pruned checkpoints
        """
        max_bytes = max_size_gb * (1024**3)
        pruned = []

        while True:
            total_size = sum(cp.size_bytes for cp in self._checkpoints)
            if total_size <= max_bytes:
                break

            # Get lowest scored non-protected checkpoint
            prunable = self._get_prunable_checkpoints()
            if not prunable:
                logger.warning("Cannot prune further - all remaining checkpoints are protected")
                break

            # Sort by score and prune lowest
            scored = [(self._score_checkpoint(cp), cp) for cp in prunable]
            scored.sort(key=lambda x: x[0])
            _, victim = scored[0]

            # Delete it
            try:
                checkpoint_path = Path(victim.path)
                if checkpoint_path.exists():
                    if checkpoint_path.is_dir():
                        shutil.rmtree(checkpoint_path)
                    else:
                        checkpoint_path.unlink()

                pruned.append(victim)
                self._checkpoints.remove(victim)
                logger.info(f"Force-pruned: run={victim.run_index}, freed={victim.size_bytes / (1024**2):.1f} MB")

            except Exception as e:
                logger.error(f"Failed to force-prune {victim.path}: {e}")
                break

        self._save_manifest()
        return pruned
