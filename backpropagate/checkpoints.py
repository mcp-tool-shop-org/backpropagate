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
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointPolicy",
    "CheckpointInfo",
    "CheckpointStats",
    "CheckpointManager",
    "RunHistoryManager",
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
        run_id: Optional correlation token (B-001) so manifest entries can
            be grepped by the same identifier used in log lines + SLAO
            merge_history.json.
    """
    run_index: int
    path: str
    validation_loss: float | None = None
    training_loss: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_run_boundary: bool = False
    is_final: bool = False
    size_bytes: int = 0
    protected: bool = False
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointInfo":
        # Forward-compat: tolerate unknown keys from newer manifests by
        # filtering down to the dataclass fields we know about.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


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
    best_checkpoint: CheckpointInfo | None = None
    oldest_checkpoint: CheckpointInfo | None = None
    newest_checkpoint: CheckpointInfo | None = None
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
        policy: CheckpointPolicy | None = None,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored
            policy: Retention policy (uses defaults if not provided)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.policy = policy or CheckpointPolicy()
        self._checkpoints: list[CheckpointInfo] = []
        self._manifest_path = self.checkpoint_dir / self.MANIFEST_FILE

        # Create directory if needed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load checkpoint manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
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

    def _save_manifest(self) -> bool:
        """Save checkpoint manifest to disk.

        Stage C BACKEND-B-006: writes are atomic. We write to a sibling
        ``.tmp`` file, fsync it, then ``os.replace`` into place. ``os.replace``
        is atomic on POSIX *and* on NTFS (Windows) — see the contract on
        https://docs.python.org/3/library/os.html#os.replace. If the process
        is killed between the open and the replace, the on-disk
        ``manifest.json`` remains the prior healthy version instead of a
        truncated JSON that ``_load_manifest`` would silently reset.

        The manifest is the **index** of every checkpoint on disk. Losing it
        because of a mid-write crash (SIGKILL, power loss, OOM-killer) causes
        the entire session's checkpoint metadata to vanish — the .pt files
        remain but the manager forgets they exist. This contract closes that
        gap.

        Returns:
            True if saved successfully, False on failure.
        """
        tmp_path = self._manifest_path.with_suffix(
            self._manifest_path.suffix + ".tmp"
        )
        try:
            data = {
                "version": "1.0",
                "updated": datetime.now().isoformat(),
                "policy": asdict(self.policy),
                "checkpoints": [c.to_dict() for c in self._checkpoints],
            }
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as fsync_err:
                    # fsync is unsupported on some filesystems / streams (rare
                    # on real disks). The atomic-replace below still preserves
                    # the prior good manifest if a crash interrupts us — we
                    # just lose the "survive a power loss after fsync" promise.
                    logger.debug(
                        f"fsync skipped (filesystem unsupported): {fsync_err}"
                    )
            os.replace(tmp_path, self._manifest_path)
            logger.debug(f"Saved manifest with {len(self._checkpoints)} checkpoints")
            return True
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            # Best-effort cleanup of the temp file so the dir stays tidy.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass  # nosec B110 — cleanup is best-effort
            return False

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

    def find_latest_for_run_id(self, run_id: str) -> CheckpointInfo | None:
        """F-002: return the most recent checkpoint tagged with ``run_id``.

        Used by the resume path: the on-disk run-history entry's run_id
        is the same correlation token written into the checkpoint
        manifest (B-001), so we can recover the last good checkpoint for
        a crashed multi-run by grepping the manifest.

        Returns ``None`` when no matching checkpoint is found.
        """
        matched = [cp for cp in self._checkpoints if cp.run_id == run_id]
        if not matched:
            return None
        # Highest run_index → most recent.
        return max(matched, key=lambda cp: cp.run_index)

    def register(
        self,
        run_index: int,
        checkpoint_path: str,
        validation_loss: float | None = None,
        training_loss: float | None = None,
        is_run_boundary: bool = False,
        protected: bool = False,
        run_id: str | None = None,
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
            run_id: Optional correlation token (B-001) persisted on the
                manifest entry so triage can grep one identifier across logs
                + checkpoints + SLAO history.

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
            run_id=run_id,
        )

        self._checkpoints.append(info)
        try:
            if not self._save_manifest():
                logger.warning(
                    "Manifest save failed after registering checkpoint — "
                    "in-memory state may diverge from disk"
                )
        except Exception as e:
            logger.warning(f"Manifest save error after registering checkpoint: {e}")

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

    def _get_prunable_checkpoints(self) -> list[CheckpointInfo]:
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

    def prune(self, dry_run: bool = False) -> list[CheckpointInfo]:
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

        try:
            if not self._save_manifest():
                logger.warning(
                    "Manifest save failed after pruning — "
                    "in-memory state may diverge from disk"
                )
        except Exception as e:
            logger.warning(f"Manifest save error after pruning: {e}")

        logger.info(
            f"Pruned {len(pruned)} checkpoints, "
            f"freed {freed_bytes / (1024**3):.2f} GB"
        )

        return pruned

    def get_best_checkpoint(self) -> CheckpointInfo | None:
        """Get the checkpoint with lowest validation loss."""
        valid = [cp for cp in self._checkpoints if cp.validation_loss is not None]
        if not valid:
            return None
        return min(valid, key=lambda x: x.validation_loss if x.validation_loss is not None else float('inf'))

    def get_final_checkpoint(self) -> CheckpointInfo | None:
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

    def list_checkpoints(self) -> list[CheckpointInfo]:
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
                try:
                    if not self._save_manifest():
                        logger.warning("Manifest save failed after protecting checkpoint")
                except Exception as e:
                    logger.warning(f"Manifest save error after protecting checkpoint: {e}")
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
                try:
                    if not self._save_manifest():
                        logger.warning("Manifest save failed after unprotecting checkpoint")
                except Exception as e:
                    logger.warning(f"Manifest save error after unprotecting checkpoint: {e}")
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
            try:
                if not self._save_manifest():
                    logger.warning("Manifest save failed after cleaning orphaned entries")
            except Exception as e:
                logger.warning(f"Manifest save error after cleaning orphaned entries: {e}")

        return len(orphaned)

    def force_prune_to_size(self, max_size_gb: float) -> list[CheckpointInfo]:
        """
        Force prune checkpoints until total size is under limit.

        The loop prunes the lowest-scored prunable checkpoint per iteration and
        stops as soon as the total size fits ``max_size_gb`` OR no prunable
        checkpoint remains. Protected checkpoints (``protected=True``), the
        final checkpoint when ``keep_final=True``, run-boundary checkpoints
        when ``keep_run_boundaries=True``, and the top-N by validation loss
        when ``keep_best_n > 0`` are all skipped — even under force prune. So
        an aggressive ``max_size_gb`` (e.g. 0.0) cannot delete the final
        checkpoint or otherwise violate the retention policy; the loop simply
        logs that it cannot prune further and exits cleanly.

        Args:
            max_size_gb: Maximum total size in gigabytes

        Returns:
            List of pruned checkpoints (may be smaller than needed if the
            remaining checkpoints are all protected by policy).
        """
        # Semantics verified during the v1.1.0 promotion swarm (Stage A) —
        # force_prune_to_size correctly delegates to _get_prunable_checkpoints,
        # which honors protected / keep_final / keep_run_boundaries / keep_best_n.
        # The previously-skipped tests/test_checkpoints.py::test_force_prune_to_size
        # passes against this implementation.
        max_bytes = max_size_gb * (1024**3)
        pruned = []
        max_iterations = 100

        for _iteration in range(max_iterations):
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
        else:
            logger.error(f"force_prune_to_size hit {max_iterations} iteration limit — aborting to prevent infinite loop")

        try:
            if not self._save_manifest():
                logger.warning("Manifest save failed after force prune")
        except Exception as e:
            logger.warning(f"Manifest save error after force prune: {e}")
        return pruned


# =============================================================================
# RUN HISTORY MANAGER
# =============================================================================

class RunHistoryManager:
    """
    Lightweight persistence layer for training run metadata.

    Stores run history as a JSON array in ``run_history.json`` inside the
    output directory.  Each entry captures the key facts about a completed
    training run so they survive process exit and can be queried later.

    Usage:
        from backpropagate.checkpoints import RunHistoryManager

        history = RunHistoryManager("/path/to/output")
        history.record_run({
            "run_id": "abc123",
            "model_name": "llama-7b",
            "final_loss": 0.42,
            ...
        })

        best = history.get_best_run()
        all_runs = history.get_history()
    """

    HISTORY_FILE = "run_history.json"

    # F-003: status taxonomy for lifecycle queries (list-runs --status).
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    VALID_STATUSES = frozenset({STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED})

    # Fields that a well-formed run entry should contain.
    _EXPECTED_FIELDS = frozenset({
        "run_id",
        "timestamp",
        "model_name",
        "dataset_info",
        "steps",
        "final_loss",
        "loss_history",
        "hyperparameters",
        "duration_seconds",
        "gpu_max_temp",
        "checkpoint_path",
        # F-003 lifecycle fields:
        "status",            # running | completed | failed
        "started_at",        # ISO timestamp at record_run_started
        "completed_at",      # ISO timestamp at record_run_completed / record_run_failed
        "session_kind",      # "single_run" | "multi_run"
        "failure_reason",    # populated only on STATUS_FAILED
        # F-004 model-card / provenance fields:
        "dataset_hash",      # sha256 hex (first 16 chars) of the dataset, when available
        "merge_history",     # list of SLAO merge_result dicts when multi_run mode
        "export_paths",      # list of export directories (populated by export.py)
    })

    # Maximum loss-history samples stored per run to bound file size.
    MAX_LOSS_HISTORY_POINTS = 100

    def __init__(self, output_dir: str) -> None:
        """
        Initialize the run history manager.

        Args:
            output_dir: Directory where ``run_history.json`` will be stored.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self.output_dir / self.HISTORY_FILE

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load(self) -> list[dict[str, Any]]:
        """Load the history file from disk, returning an empty list on failure."""
        if not self._history_path.exists():
            return []
        try:
            with open(self._history_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("run_history.json is not a JSON array — resetting")
            return []
        except Exception as e:
            logger.warning(f"Failed to load run history: {e}")
            return []

    def _save(self, history: list[dict[str, Any]]) -> bool:
        """Persist the history list to disk.

        Stage C BACKEND-B-006: atomic write (tmp + fsync + os.replace) so a
        mid-write crash doesn't strand the on-disk ``run_history.json`` in a
        truncated JSON state. ``_load`` falls back to ``[]`` on JSONDecodeError,
        which would silently wipe the operator's session history. The atomic
        contract preserves the last known-good version through SIGKILL / power
        loss / OOM-killer. See companion contract on ``_save_manifest``.

        Returns:
            True on success, False on failure.
        """
        tmp_path = self._history_path.with_suffix(
            self._history_path.suffix + ".tmp"
        )
        try:
            with open(tmp_path, "w") as f:
                json.dump(history, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as fsync_err:
                    logger.debug(
                        f"fsync skipped (filesystem unsupported): {fsync_err}"
                    )
            os.replace(tmp_path, self._history_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save run history: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass  # nosec B110 — cleanup is best-effort
            return False

    @staticmethod
    def _downsample_loss_history(
        loss_history: list[float],
        max_points: int,
    ) -> list[float]:
        """Downsample a loss history list to at most *max_points* entries.

        Uses uniform index sampling so the shape of the curve is preserved.
        """
        if len(loss_history) <= max_points:
            return list(loss_history)

        step = len(loss_history) / max_points
        return [
            loss_history[int(i * step)]
            for i in range(max_points)
        ]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def record_run(self, run_data: dict[str, Any]) -> dict[str, Any]:
        """
        Append a completed run to the history file.

        The caller provides a dict with run metadata.  Missing fields are
        filled with ``None`` so every entry has a consistent shape.  The
        ``loss_history`` list is downsampled to at most
        :pyattr:`MAX_LOSS_HISTORY_POINTS` entries.

        Args:
            run_data: Dictionary of run metadata.  Expected keys:
                run_id, timestamp, model_name, dataset_info, steps,
                final_loss, loss_history, hyperparameters,
                duration_seconds, gpu_max_temp, checkpoint_path.

        Returns:
            The normalised entry that was persisted.
        """
        # Build a normalised entry with every expected field.
        entry: dict[str, Any] = {}
        for key in self._EXPECTED_FIELDS:
            entry[key] = run_data.get(key)

        # Ensure a timestamp exists.
        if entry.get("timestamp") is None:
            entry["timestamp"] = datetime.now().isoformat()

        # Downsample loss_history to bound file size.
        if isinstance(entry.get("loss_history"), list):
            entry["loss_history"] = self._downsample_loss_history(
                entry["loss_history"],
                self.MAX_LOSS_HISTORY_POINTS,
            )

        # Persist.
        history = self._load()
        history.append(entry)
        if not self._save(history):
            logger.warning(
                "Run history save failed — entry may be lost on next load"
            )

        logger.info(
            f"Recorded run: id={entry.get('run_id')}, "
            f"final_loss={entry.get('final_loss')}, "
            f"steps={entry.get('steps')}"
        )
        return entry

    def get_history(self) -> list[dict[str, Any]]:
        """
        Load and return the full run history.

        Returns:
            List of run-metadata dicts, oldest first.
        """
        return self._load()

    def get_best_run(self) -> dict[str, Any] | None:
        """
        Return the run with the lowest ``final_loss``.

        Runs where ``final_loss`` is ``None`` are skipped.

        Returns:
            The best run dict, or ``None`` if no runs have a recorded loss.
        """
        history = self._load()
        valid = [r for r in history if r.get("final_loss") is not None]
        if not valid:
            return None
        return min(valid, key=lambda r: r["final_loss"])

    # --------------------------------------------------------------------- #
    # F-003 lifecycle API
    # --------------------------------------------------------------------- #
    # The lifecycle API was added to wire RunHistoryManager into Trainer +
    # MultiRunTrainer so that every training session leaves a queryable record
    # on disk. The flow is:
    #
    #   record_run_started(run_id, ...) -> entry with status="running"
    #   record_run_completed(run_id, ...) -> updates status to "completed"
    #   record_run_failed(run_id, reason) -> updates status to "failed"
    #
    # The lookup helpers (get_run / list_runs / delete_run) back the CLI
    # ``backprop list-runs`` and ``backprop show-run`` subcommands.

    def record_run_started(
        self,
        run_id: str,
        model_name: str | None = None,
        dataset_info: Any = None,
        hyperparameters: dict[str, Any] | None = None,
        session_kind: str = "single_run",
        checkpoint_path: str | None = None,
        dataset_hash: str | None = None,
    ) -> dict[str, Any]:
        """Record the start of a training run (status="running").

        Each entry is keyed by ``run_id``; calling ``record_run_started``
        twice with the same run_id updates the existing entry in place
        rather than creating a duplicate. This is the foundation for the
        F-002 resume flow (an in-progress entry signals "this run can be
        resumed from its checkpoint").
        """
        now = datetime.now().isoformat()
        entry: dict[str, Any] = dict.fromkeys(self._EXPECTED_FIELDS)
        entry.update({
            "run_id": run_id,
            "timestamp": now,
            "started_at": now,
            "status": self.STATUS_RUNNING,
            "session_kind": session_kind,
            "model_name": model_name,
            "dataset_info": dataset_info,
            "dataset_hash": dataset_hash,
            "hyperparameters": hyperparameters or {},
            "checkpoint_path": checkpoint_path,
            "loss_history": [],
            "merge_history": [],
            "export_paths": [],
        })

        history = self._load()
        # Replace any existing entry with the same run_id (idempotent start).
        history = [r for r in history if r.get("run_id") != run_id]
        history.append(entry)
        if not self._save(history):
            logger.warning("record_run_started: save failed")

        logger.info(
            f"Recorded run start: run_id={run_id} model={model_name} "
            f"session_kind={session_kind}"
        )
        return entry

    def record_run_completed(
        self,
        run_id: str,
        final_loss: float | None = None,
        loss_history: list[float] | None = None,
        steps: int | None = None,
        duration_seconds: float | None = None,
        gpu_max_temp: float | None = None,
        checkpoint_path: str | None = None,
        merge_history: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark a previously-started run as completed and persist final metrics."""
        history = self._load()
        matched = False
        now = datetime.now().isoformat()
        for entry in history:
            if entry.get("run_id") != run_id:
                continue
            matched = True
            entry["status"] = self.STATUS_COMPLETED
            entry["completed_at"] = now
            if final_loss is not None:
                entry["final_loss"] = final_loss
            if loss_history is not None:
                entry["loss_history"] = self._downsample_loss_history(
                    loss_history,
                    self.MAX_LOSS_HISTORY_POINTS,
                )
            if steps is not None:
                entry["steps"] = steps
            if duration_seconds is not None:
                entry["duration_seconds"] = duration_seconds
            if gpu_max_temp is not None:
                entry["gpu_max_temp"] = gpu_max_temp
            if checkpoint_path is not None:
                entry["checkpoint_path"] = checkpoint_path
            if merge_history is not None:
                entry["merge_history"] = merge_history
            if extra:
                for key, value in extra.items():
                    entry[key] = value
            break

        if not matched:
            # No started-record found — synthesize one so we don't silently
            # drop the completion signal. This can happen if the caller
            # forgot record_run_started, or if record_run_started's save
            # failed.
            logger.warning(
                f"record_run_completed: no started record for run_id={run_id}; "
                "synthesizing entry"
            )
            entry = dict.fromkeys(self._EXPECTED_FIELDS)
            entry.update({
                "run_id": run_id,
                "timestamp": now,
                "completed_at": now,
                "status": self.STATUS_COMPLETED,
                "final_loss": final_loss,
                "loss_history": self._downsample_loss_history(
                    loss_history or [],
                    self.MAX_LOSS_HISTORY_POINTS,
                ),
                "steps": steps,
                "duration_seconds": duration_seconds,
                "gpu_max_temp": gpu_max_temp,
                "checkpoint_path": checkpoint_path,
                "merge_history": merge_history or [],
                "export_paths": [],
            })
            if extra:
                entry.update(extra)
            history.append(entry)

        if not self._save(history):
            logger.warning("record_run_completed: save failed")
        logger.info(
            f"Recorded run completed: run_id={run_id} final_loss={final_loss}"
        )
        return entry

    def record_run_failed(
        self,
        run_id: str,
        failure_reason: str,
        loss_history: list[float] | None = None,
        duration_seconds: float | None = None,
        checkpoint_path: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark a previously-started run as failed and persist the failure reason."""
        history = self._load()
        matched = False
        now = datetime.now().isoformat()
        for entry in history:
            if entry.get("run_id") != run_id:
                continue
            matched = True
            entry["status"] = self.STATUS_FAILED
            entry["completed_at"] = now
            entry["failure_reason"] = failure_reason
            if loss_history is not None:
                entry["loss_history"] = self._downsample_loss_history(
                    loss_history,
                    self.MAX_LOSS_HISTORY_POINTS,
                )
            if duration_seconds is not None:
                entry["duration_seconds"] = duration_seconds
            if checkpoint_path is not None:
                entry["checkpoint_path"] = checkpoint_path
            if extra:
                for key, value in extra.items():
                    entry[key] = value
            break

        if not matched:
            logger.warning(
                f"record_run_failed: no started record for run_id={run_id}; "
                "synthesizing entry"
            )
            entry = dict.fromkeys(self._EXPECTED_FIELDS)
            entry.update({
                "run_id": run_id,
                "timestamp": now,
                "completed_at": now,
                "status": self.STATUS_FAILED,
                "failure_reason": failure_reason,
                "loss_history": self._downsample_loss_history(
                    loss_history or [],
                    self.MAX_LOSS_HISTORY_POINTS,
                ),
                "duration_seconds": duration_seconds,
                "checkpoint_path": checkpoint_path,
                "merge_history": [],
                "export_paths": [],
            })
            if extra:
                entry.update(extra)
            history.append(entry)

        if not self._save(history):
            logger.warning("record_run_failed: save failed")
        logger.info(
            f"Recorded run failed: run_id={run_id} reason={failure_reason}"
        )
        return entry

    def update_run(
        self,
        run_id: str,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Patch fields on an existing run entry.

        Returns the updated entry, or ``None`` if the run_id was not found.
        Used by export.py to append to ``export_paths`` and by the resume
        path to roll ``status`` from "failed" back to "running".
        """
        if not fields:
            return self.get_run(run_id)

        history = self._load()
        updated: dict[str, Any] | None = None
        for entry in history:
            if entry.get("run_id") != run_id:
                continue
            for key, value in fields.items():
                entry[key] = value
            updated = entry
            break

        if updated is None:
            logger.warning(f"update_run: run_id={run_id} not found")
            return None

        if not self._save(history):
            logger.warning("update_run: save failed")
        return updated

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the run entry for ``run_id`` or ``None`` if not present.

        Supports partial-prefix matching for operator convenience: if no
        exact match is found and ``run_id`` is a strict prefix of exactly
        one entry's run_id, return that entry. (CLI users typically only
        type the first 8-12 hex chars.)
        """
        history = self._load()
        for entry in history:
            if entry.get("run_id") == run_id:
                return entry
        # Partial-prefix fallback.
        prefix_matches = [
            entry for entry in history
            if isinstance(entry.get("run_id"), str)
            and entry["run_id"].startswith(run_id)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def list_runs(
        self,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return all run entries, most recent first.

        Args:
            status: Optional filter — one of ``running`` / ``completed`` /
                ``failed``. Invalid values raise ``ValueError``.
            limit: Optional cap on the number of entries returned.

        Returns:
            List of run dicts, newest-first (by ``started_at`` / ``timestamp``).
        """
        if status is not None and status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Expected one of: "
                f"{sorted(self.VALID_STATUSES)}"
            )

        history = self._load()
        if status is not None:
            history = [r for r in history if r.get("status") == status]

        # Sort newest-first using the best timestamp we have. ``started_at``
        # is the lifecycle-API field; pre-lifecycle entries only carry
        # ``timestamp``. Use whichever is present.
        def _sort_key(entry: dict[str, Any]) -> str:
            return str(
                entry.get("started_at")
                or entry.get("timestamp")
                or ""
            )

        history.sort(key=_sort_key, reverse=True)

        if limit is not None and limit > 0:
            history = history[:limit]
        return history

    def delete_run(self, run_id: str) -> bool:
        """Remove a run entry by ``run_id``. Returns True on success."""
        history = self._load()
        before = len(history)
        history = [r for r in history if r.get("run_id") != run_id]
        if len(history) == before:
            logger.warning(f"delete_run: run_id={run_id} not found")
            return False
        if not self._save(history):
            logger.warning("delete_run: save failed")
            return False
        logger.info(f"Deleted run history entry: run_id={run_id}")
        return True

    def in_progress_runs(self) -> list[dict[str, Any]]:
        """Return runs that are currently in the ``running`` state.

        Used by the F-002 resume auto-detect path so a crashed multi-run
        can be picked up without an explicit ``--resume <run-id>``.
        """
        return [r for r in self._load() if r.get("status") == self.STATUS_RUNNING]
