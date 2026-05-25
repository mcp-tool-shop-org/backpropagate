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

import contextlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# BACKEND-F-012 (Wave 6a): cross-platform advisory file locking around
# RunHistoryManager mutators. ``filelock`` is a transitive dep via every
# core dep (huggingface_hub, datasets, torch, transformers all require
# it); it is also added explicitly in pyproject.toml core deps so the
# import is guaranteed regardless of upstream pin shifts. The fallback
# below preserves the lock-less behavior if a future stripped install
# drops ``filelock`` from the env — load-bearing for the no-op upgrade
# path (existing single-operator-per-output_dir setups keep working).
try:
    from filelock import FileLock
    from filelock import Timeout as FileLockTimeout

    _FILELOCK_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only in stripped installs
    FileLock = None  # type: ignore[assignment, misc]
    FileLockTimeout = TimeoutError  # type: ignore[assignment, misc]
    _FILELOCK_AVAILABLE = False

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

    # Stage C amend BACKEND-B-003: schema version anchor for the manifest.
    # Bump on incompatible changes (e.g. field rename or semantic shift);
    # leave at "1.0" for additive fields (those are forward-compat via
    # ``CheckpointInfo.from_dict``'s unknown-key filter). v1.4 may build a
    # real migrator; v1.3 just fails-loud-but-keeps-going on mismatch.
    CURRENT_MANIFEST_VERSION = "1.0"

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
        """Load checkpoint manifest from disk.

        Stage C amend BACKEND-B-003: validate the on-disk ``version`` field
        against :attr:`CURRENT_MANIFEST_VERSION` and WARN on mismatch.
        ``CheckpointInfo.from_dict`` already discards unknown keys
        (forward-compat for additive schema changes); the version check
        adds backward-compat fail-loud-but-keep-going semantics. v1.4 will
        ship a real migrator. Missing version (pre-v1.0 manifests) defaults
        to "0.0" and surfaces the warn line so operators correlate
        post-resume behavior with schema age.
        """
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    data = json.load(f)
                disk_version = str(data.get("version") or "0.0")
                if disk_version != self.CURRENT_MANIFEST_VERSION:
                    logger.warning(
                        f"Checkpoint manifest on disk has "
                        f"version={disk_version!r} but this build expects "
                        f"{self.CURRENT_MANIFEST_VERSION!r}. Unknown fields "
                        f"will be discarded by CheckpointInfo.from_dict; "
                        f"missing fields fall back to defaults. v1.4 will "
                        f"ship a real migrator for older formats."
                    )
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
            # Stage C humanization: name what the failure does and does
            # not affect. The manifest is best-effort metadata; the
            # on-disk checkpoints themselves (PEFT directories) are
            # unaffected. Operator next-step is to verify free disk
            # space + write permission on checkpoint_dir.
            logger.error(
                "Failed to save checkpoint manifest at %s: %s. "
                "On-disk checkpoint directories are unaffected; only the "
                "resume-candidate index is missing. Verify free disk "
                "space and write permission on %s, then run the next "
                "save to re-populate the manifest.",
                self._manifest_path,
                e,
                self._manifest_path.parent,
            )
            # Best-effort cleanup of the temp file so the dir stays tidy.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
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
                # Stage C humanization: name what's blocking the prune so
                # the operator can decide whether to relax the policy or
                # raise the size cap. The protected set is the union of
                # keep_final + keep_run_boundaries + keep_best_n +
                # explicitly-protected entries; pre-fix the operator had
                # to read the source to know what "protected" meant.
                logger.warning(
                    "Cannot prune further: all %d remaining checkpoints "
                    "are protected by CheckpointPolicy (keep_final=%s, "
                    "keep_run_boundaries=%s, keep_best_n=%d, plus any "
                    "explicitly-protected entries). Total size %.1f GB "
                    "exceeds max_size_gb=%.1f. Options: (1) raise "
                    "max_size_gb, (2) lower keep_best_n / disable "
                    "keep_run_boundaries in CheckpointPolicy, or (3) "
                    "manually unprotect specific checkpoints via "
                    "CheckpointManager.unprotect(<path>).",
                    len(self._checkpoints),
                    self.policy.keep_final,
                    self.policy.keep_run_boundaries,
                    self.policy.keep_best_n,
                    sum(cp.size_bytes for cp in self._checkpoints) / (1024**3),
                    max_size_gb,
                )
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
            # Stage C amend BACKEND-B-022: include actionable context so an
            # operator who hits the safety guard has the info they need to
            # decide between "raise the limit" and "file a bug." Pre-fix
            # the log line stopped at "aborting" with no diagnostics.
            current_total_bytes = sum(cp.size_bytes for cp in self._checkpoints)
            current_total_gb = current_total_bytes / (1024**3)
            logger.error(
                f"force_prune_to_size hit {max_iterations} iteration limit "
                f"— aborting to prevent infinite loop. State at abort: "
                f"checkpoints_remaining={len(self._checkpoints)}, "
                f"total_size_gb={current_total_gb:.2f}, "
                f"target_size_gb={max_size_gb:.2f}, "
                f"pruned_this_pass={len(pruned)}. "
                f"If this is a legitimate large-session, raise "
                f"force_prune_to_size's max_iterations ceiling (current "
                f"hard-coded at {max_iterations}); if not, file a bug at "
                f"https://github.com/mcp-tool-shop-org/backpropagate/issues "
                f"with the manifest.json + this log line."
            )

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

    # Stage C amend BACKEND-B-003: per-entry schema version. The file
    # itself is a JSON array (changing that shape would break operators
    # mid-flight in v1.3), so we anchor versioning at the entry level.
    # Bump on incompatible per-entry shape changes; leave at "1.0" for
    # additive fields (record_run_* tolerates missing optional fields).
    # v1.4 may introduce a per-file envelope; for now the per-entry tag is
    # enough to flag mismatched-version writers in a mixed-tooling shop.
    CURRENT_ENTRY_SCHEMA_VERSION = "1.0"

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

    def __init__(
        self,
        output_dir: str,
        on_record_callback: Callable[[dict[str, Any]], None] | None = None,
        lock_timeout_seconds: float | None = None,
    ) -> None:
        """
        Initialize the run history manager.

        Args:
            output_dir: Directory where ``run_history.json`` will be stored.
            on_record_callback: Stage C amend BACKEND-B-016 — optional
                hook called after every record_run / record_run_started /
                record_run_completed / record_run_failed with the
                normalized entry dict. Use this to ship records to
                external systems (DataDog, ELK, wandb, mlflow, custom
                DB) without wrapping every train() call yourself.
                Callback failures are swallowed (logged at WARN) so a
                broken sink never breaks on-disk persistence. The
                callback is invoked OUTSIDE the on-disk save so it can
                run in parallel with the next write — but it IS
                synchronous; long-running sinks should themselves spawn
                a worker thread / queue.
            lock_timeout_seconds: BACKEND-F-012 (Wave 6a) — maximum wait
                for the cross-platform ``filelock`` advisory lock around
                the load+mutate+save cycle in every public mutator
                (``record_run`` / ``record_run_started`` /
                ``record_run_completed`` / ``record_run_failed`` /
                ``update_run`` / ``delete_run``). When two operators run
                ``backprop train`` against the same output_dir
                simultaneously, the lock serializes their writes so
                neither's entry is silently dropped and ``run_history.json``
                never lands in a truncated mid-write state from two
                interleaved ``json.dump`` calls on the same .tmp file.
                Defaults to :attr:`DEFAULT_LOCK_TIMEOUT_SECONDS` (30s).
                Pass 0 to disable the timeout (block forever) — useful
                for orchestrators that prefer queuing over failing fast.
                On timeout, mutators log a structured error and return
                False rather than raising, so a stuck holder never
                cascades into a hard training failure (history
                persistence is best-effort by Stage C contract).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self.output_dir / self.HISTORY_FILE
        self._on_record_callback = on_record_callback
        # BACKEND-F-012 (Wave 6a): lock-file path is a sibling of the
        # history file. ``.lock`` suffix is conventional for ``filelock``
        # and lets the file be safely gitignored / cleaned-up by
        # housekeeping tooling without touching the source-of-truth JSON.
        self._lock_path = self.output_dir / f"{self.HISTORY_FILE}.lock"
        self._lock_timeout_seconds = (
            self.DEFAULT_LOCK_TIMEOUT_SECONDS
            if lock_timeout_seconds is None
            else float(lock_timeout_seconds)
        )

    def _fire_callback(self, entry: dict[str, Any]) -> None:
        """Stage C amend BACKEND-B-016: invoke the optional record hook,
        swallowing any exception so a broken sink never breaks on-disk
        persistence. The error is logged at WARN with the callback's
        repr so the operator can correlate failures to a specific sink.
        """
        if self._on_record_callback is None:
            return
        try:
            self._on_record_callback(entry)
        except Exception as cb_err:
            logger.warning(
                f"RunHistoryManager on_record_callback raised "
                f"({type(cb_err).__name__}: {cb_err}); on-disk persistence "
                f"is unaffected but the external sink missed this entry."
            )

    @contextlib.contextmanager
    def _locked_mutate(self, operation: str) -> Iterator[bool]:
        """BACKEND-F-012 (Wave 6a): cross-platform file lock around the
        load+mutate+save cycle in every public mutator.

        Yields True when the lock was acquired (caller proceeds with the
        critical section); yields False when acquisition timed out or
        ``filelock`` is unavailable in a degraded environment (caller
        proceeds WITHOUT serialization — preserves the prior behavior
        rather than failing the training run because of a stuck holder).

        The history-persistence contract is best-effort (see Stage C
        BACKEND-B-006: ``_save`` returns False on failure rather than
        raising, and every caller logs WARN and continues). The lock
        timeout follows the same contract: a stuck holder produces a
        diagnosable log line but never cascades into a hard training
        failure. Two concurrent ``backprop train`` invocations that race
        on this lock will still serialize correctly via the underlying
        OS file-lock; the timeout protects against a *crashed* holder
        whose lock the OS hasn't released (rare on POSIX, possible on
        Windows when a process is force-killed mid-critical-section).

        Args:
            operation: Short tag used in log lines so an operator can
                trace which mutator hit the timeout
                (e.g. "record_run_started", "delete_run"). Not parsed —
                purely cosmetic for triage.
        """
        if not _FILELOCK_AVAILABLE or FileLock is None:
            # Degraded fallback: no lock available. Log at DEBUG (not
            # WARN — a stripped install is a deliberate operator choice
            # and the prior unbounded behavior was the baseline) and
            # proceed without serialization.
            logger.debug(
                f"RunHistoryManager._locked_mutate({operation}): filelock "
                f"unavailable; proceeding without cross-process serialization. "
                f"Concurrent writes from a second process may race."
            )
            yield True
            return

        # ``filelock`` interprets timeout=-1 as block-forever. Our public
        # contract is "0 disables the timeout" so we translate here.
        effective_timeout = self._lock_timeout_seconds
        if effective_timeout <= 0:
            effective_timeout = -1

        lock = FileLock(str(self._lock_path), timeout=effective_timeout)
        try:
            with lock:
                yield True
        except FileLockTimeout:
            # A stuck holder — likely a crashed process whose lock file
            # the OS hasn't released. Log structured + actionable so the
            # operator can decide whether to delete the .lock file (safe
            # ONLY when they know no live process holds it).
            logger.error(
                f"RunHistoryManager._locked_mutate({operation}): lock "
                f"acquisition timed out after {self._lock_timeout_seconds:.1f}s "
                f"on {self._lock_path!s}. A concurrent writer is either still "
                f"holding the lock (legitimate contention) or crashed mid-"
                f"critical-section (stuck lock). If you have verified no live "
                f"process is writing to {self._history_path!s}, manually "
                f"removing {self._lock_path!s} will clear the stuck holder. "
                f"Skipping this mutation to avoid blocking the training run."
            )
            yield False

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load(self) -> list[dict[str, Any]]:
        """Load the history file from disk, returning an empty list on failure.

        Stage C amend BACKEND-B-003: scan loaded entries for unfamiliar
        ``schema_version`` values and WARN once per unique mismatch. The
        scan is non-fatal: forward-compat is best-effort via field
        defaults, backward-compat (entries lacking ``schema_version``) is
        the "0.0" implicit baseline. v1.4 may build a real migrator.
        """
        if not self._history_path.exists():
            return []
        try:
            with open(self._history_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                # Stage C amend BACKEND-B-003: surface schema-version
                # mismatches once per unique value so a mixed-version
                # writer (e.g. v1.2 wrote a row, v1.3 reads it) is
                # diagnosable from a single log line.
                seen_versions: set[str] = set()
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    ver = str(entry.get("schema_version") or "0.0")
                    if ver != self.CURRENT_ENTRY_SCHEMA_VERSION and ver not in seen_versions:
                        seen_versions.add(ver)
                        logger.warning(
                            f"run_history.json contains entry with "
                            f"schema_version={ver!r} but this build expects "
                            f"{self.CURRENT_ENTRY_SCHEMA_VERSION!r}. Missing "
                            f"fields will fall back to defaults — re-record "
                            f"the entry under v1.3 to clear this warning."
                        )
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
                pass
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

        # Stage C amend BACKEND-B-003: anchor schema version on every
        # entry we write. Preserve any version the caller passed in
        # explicitly (e.g. test fixtures that simulate older formats).
        if not entry.get("schema_version"):
            entry["schema_version"] = self.CURRENT_ENTRY_SCHEMA_VERSION

        # Ensure a timestamp exists.
        if entry.get("timestamp") is None:
            entry["timestamp"] = datetime.now().isoformat()

        # Downsample loss_history to bound file size.
        if isinstance(entry.get("loss_history"), list):
            entry["loss_history"] = self._downsample_loss_history(
                entry["loss_history"],
                self.MAX_LOSS_HISTORY_POINTS,
            )

        # Persist. BACKEND-F-012 (Wave 6a): the load+mutate+save cycle
        # runs under the cross-platform file lock so a concurrent
        # ``backprop train`` against the same output_dir cannot
        # interleave-and-overwrite this entry. On stuck-lock timeout
        # the helper logs structured + actionable and yields False;
        # we still attempt the write (degraded to prior unlocked
        # behavior) so a crashed sibling never strands fresh entries.
        with self._locked_mutate("record_run"):
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
        # Stage C amend BACKEND-B-016: fire the external-sink callback
        # after on-disk persistence so a broken sink doesn't strand the
        # in-memory entry.
        self._fire_callback(entry)
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
            # Stage C amend BACKEND-B-003: anchor schema version on every
            # entry we write so a future reader can detect mismatch.
            "schema_version": self.CURRENT_ENTRY_SCHEMA_VERSION,
        })

        # BACKEND-F-012 (Wave 6a): load+mutate+save under file lock so a
        # concurrent ``backprop train`` start against the same output_dir
        # cannot interleave with the idempotent replace below.
        with self._locked_mutate("record_run_started"):
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
        # Stage C amend BACKEND-B-016: fire the external-sink callback.
        self._fire_callback(entry)
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
        """Mark a previously-started run as completed and persist final metrics.

        BACKEND-F-012 (Wave 6a): the load+mutate+save cycle runs under
        the cross-platform file lock so a concurrent writer (another
        ``backprop train`` against the same output_dir, or a multi-run
        session firing record_run_completed simultaneously) cannot
        clobber this update.
        """
        with self._locked_mutate("record_run_completed"):
            entry = self._mutate_completed(
                run_id=run_id,
                final_loss=final_loss,
                loss_history=loss_history,
                steps=steps,
                duration_seconds=duration_seconds,
                gpu_max_temp=gpu_max_temp,
                checkpoint_path=checkpoint_path,
                merge_history=merge_history,
                extra=extra,
            )
        logger.info(
            f"Recorded run completed: run_id={run_id} final_loss={final_loss}"
        )
        # Stage C amend BACKEND-B-016: fire the external-sink callback.
        self._fire_callback(entry)
        return entry

    def _mutate_completed(
        self,
        run_id: str,
        final_loss: float | None,
        loss_history: list[float] | None,
        steps: int | None,
        duration_seconds: float | None,
        gpu_max_temp: float | None,
        checkpoint_path: str | None,
        merge_history: list[dict[str, Any]] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """BACKEND-F-012 (Wave 6a): inner critical section for
        record_run_completed. Extracted so the lock context wraps the
        load+mutate+save cycle in one place without the public
        signature having to change.
        """
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
                # Stage C amend BACKEND-B-003: anchor schema version on
                # synthesized entries too.
                "schema_version": self.CURRENT_ENTRY_SCHEMA_VERSION,
            })
            if extra:
                entry.update(extra)
            history.append(entry)

        if not self._save(history):
            logger.warning("record_run_completed: save failed")
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
        """Mark a previously-started run as failed and persist the failure reason.

        BACKEND-F-012 (Wave 6a): the load+mutate+save cycle runs under
        the cross-platform file lock so a concurrent writer cannot
        clobber this update.
        """
        with self._locked_mutate("record_run_failed"):
            entry = self._mutate_failed(
                run_id=run_id,
                failure_reason=failure_reason,
                loss_history=loss_history,
                duration_seconds=duration_seconds,
                checkpoint_path=checkpoint_path,
                extra=extra,
            )
        logger.info(
            f"Recorded run failed: run_id={run_id} reason={failure_reason}"
        )
        # Stage C amend BACKEND-B-016: fire the external-sink callback.
        self._fire_callback(entry)
        return entry

    def _mutate_failed(
        self,
        run_id: str,
        failure_reason: str,
        loss_history: list[float] | None,
        duration_seconds: float | None,
        checkpoint_path: str | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """BACKEND-F-012 (Wave 6a): inner critical section for
        record_run_failed. Extracted so the lock context wraps the
        load+mutate+save cycle in one place without the public
        signature having to change.
        """
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
                # Stage C amend BACKEND-B-003: anchor schema version on
                # synthesized entries too.
                "schema_version": self.CURRENT_ENTRY_SCHEMA_VERSION,
            })
            if extra:
                entry.update(extra)
            history.append(entry)

        if not self._save(history):
            logger.warning("record_run_failed: save failed")
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

        BACKEND-F-012 (Wave 6a): the load+mutate+save cycle runs under
        the cross-platform file lock so a concurrent writer cannot
        interleave with this patch.
        """
        if not fields:
            return self.get_run(run_id)

        with self._locked_mutate("update_run"):
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
        """Remove a run entry by ``run_id``. Returns True on success.

        BACKEND-F-012 (Wave 6a): the load+mutate+save cycle runs under
        the cross-platform file lock so a concurrent writer cannot
        clobber this delete.
        """
        with self._locked_mutate("delete_run"):
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

    # Stage C amend BACKEND-B-006: default stale-threshold for in-progress
    # entries. Sessions SIGKILLed before ``record_run_failed`` could fire
    # leave ``status='running'`` on disk forever; the F-002 auto-resume
    # path would otherwise pick up months-old orphans every time the
    # operator started a fresh session in the same output dir. 24h is a
    # generous floor — operators with genuinely long-running sessions can
    # raise it via the ``stale_threshold_seconds`` kwarg.
    DEFAULT_IN_PROGRESS_TTL_SECONDS: float = 24 * 60 * 60  # 24 hours

    # BACKEND-F-012 (Wave 6a): default lock acquisition timeout for the
    # ``_locked_mutate`` cross-platform file lock around mutators. Two
    # concurrent ``backprop train`` invocations against the same output_dir
    # race on ``run_history.json``: both load → both mutate → both save,
    # last writer wins, the earlier writer's entry is silently discarded
    # AND under unlucky timing the on-disk JSON ends up truncated mid-write
    # because two ``json.dump`` calls interleave on the same .tmp file.
    # The lock serializes the whole load+mutate+save cycle. 30s is a
    # generous floor — the per-mutation critical section is sub-millisecond
    # in practice (single JSON read + list append + JSON write of ≤MAX_LOSS
    # entries); anything approaching the floor signals a stuck holder
    # (crashed process whose lock file the OS hasn't released) and the
    # operator should triage rather than wait quietly. Override via
    # ``lock_timeout_seconds`` constructor kwarg.
    DEFAULT_LOCK_TIMEOUT_SECONDS: float = 30.0

    def in_progress_runs(
        self,
        stale_threshold_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return runs that are currently in the ``running`` state.

        Used by the F-002 resume auto-detect path so a crashed multi-run
        can be picked up without an explicit ``--resume <run-id>``.

        Stage C amend BACKEND-B-006: filter out stale entries whose
        ``started_at`` is older than ``stale_threshold_seconds`` (default
        24h). The pre-fix behavior auto-resumed any ``status='running'``
        entry regardless of age, which produced confusing "No checkpoint
        found for run_id=X" warnings when an operator's NEXT fresh session
        unexpectedly latched onto a months-old orphan from a crashed
        session whose checkpoints had since been cleaned up.

        Args:
            stale_threshold_seconds: Maximum age (in seconds) for an entry
                to be considered live. Entries older than this are SKIPPED
                from the returned list AND logged so the operator sees what
                was filtered. Pass 0.0 to disable the filter entirely
                (prior unbounded behavior). ``None`` uses
                :attr:`DEFAULT_IN_PROGRESS_TTL_SECONDS`.

        Returns:
            List of run dicts in ``status='running'`` whose ``started_at``
            is within the TTL.
        """
        from datetime import datetime as _dt

        if stale_threshold_seconds is None:
            stale_threshold_seconds = self.DEFAULT_IN_PROGRESS_TTL_SECONDS

        running = [
            r for r in self._load()
            if r.get("status") == self.STATUS_RUNNING
        ]

        if stale_threshold_seconds <= 0.0 or not running:
            return running

        now = _dt.now()
        live: list[dict[str, Any]] = []
        skipped = 0
        for entry in running:
            ts = entry.get("started_at") or entry.get("timestamp")
            if not ts:
                # No timestamp anchor — keep it (pre-lifecycle entries
                # predate ``started_at``; better to surface than to drop).
                live.append(entry)
                continue
            try:
                parsed = _dt.fromisoformat(str(ts))
            except (ValueError, TypeError):
                # Unparseable timestamp — keep so the operator can triage
                # rather than silently dropping a real session.
                live.append(entry)
                continue
            age_seconds = (now - parsed).total_seconds()
            if age_seconds <= stale_threshold_seconds:
                live.append(entry)
            else:
                skipped += 1
                logger.warning(
                    f"in_progress_runs: skipping stale entry "
                    f"run_id={entry.get('run_id')!r} age={age_seconds:.0f}s "
                    f"(threshold={stale_threshold_seconds:.0f}s); the "
                    f"process_id is unknown to this OS and the checkpoint "
                    f"on disk may have been cleaned up. Re-run with "
                    f"resume_from=<run-id> to force-pick this entry, OR "
                    f"call ``delete_run({entry.get('run_id')!r})`` to clear it."
                )
        if skipped:
            logger.info(
                f"in_progress_runs: filtered {skipped} stale entries "
                f"(threshold={stale_threshold_seconds:.0f}s); "
                f"{len(live)} live."
            )
        return live
