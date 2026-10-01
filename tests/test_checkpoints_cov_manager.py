"""Coverage tests for ``CheckpointManager`` (backpropagate/checkpoints.py).

Real filesystem (``tmp_path``) and the real ``filelock`` library throughout.
Mock boundary: OS-level failure injection only (``os.replace`` / ``os.fsync`` /
``shutil.rmtree`` / ``Path.unlink`` raising), which is how disk-full, locked
file and partial-delete conditions are produced portably. Lock contention is
real: the test holds the manager's own ``.lock`` file from a second
``FileLock`` handle.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pytest
from filelock import FileLock

from backpropagate import checkpoints as ckpt_mod
from backpropagate.checkpoints import (
    CheckpointInfo,
    CheckpointManager,
    CheckpointPolicy,
    CheckpointStats,
)

LOGGER = "backpropagate.checkpoints"


def make_ckpt(base: Path, name: str, size: int = 16, *, as_file: bool = False) -> Path:
    p = base / name
    if as_file:
        p.write_bytes(b"x" * size)
    else:
        p.mkdir()
        (p / "adapter_model.bin").write_bytes(b"x" * size)
    return p


@pytest.fixture
def mgr(tmp_path):
    """Manager with auto-prune off so each test drives pruning explicitly."""
    return CheckpointManager(
        str(tmp_path / "ckpts"),
        CheckpointPolicy(keep_best_n=1, keep_final=False, keep_run_boundaries=False,
                         max_total=0, auto_prune=False),
    )


def manifest_on_disk(m: CheckpointManager) -> dict:
    return json.loads(m._manifest_path.read_text())


def fail_replace(monkeypatch, *, times: int | None = None):
    """Make ``os.replace`` (the atomic manifest swap) raise OSError.

    ``times=None`` fails every call; otherwise only the first ``times`` calls.
    """
    real = os.replace
    state = {"n": 0}

    def fake(src, dst):
        state["n"] += 1
        if times is None or state["n"] <= times:
            raise OSError("disk full")
        return real(src, dst)

    monkeypatch.setattr(ckpt_mod.os, "replace", fake)
    return state


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class TestManifestLocking:
    def test_without_filelock_register_still_persists(self, mgr, monkeypatch, caplog):
        monkeypatch.setattr(ckpt_mod, "_FILELOCK_AVAILABLE", False)
        p = make_ckpt(mgr.checkpoint_dir, "c1")
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            info = mgr.register(0, str(p), validation_loss=0.5)
        assert info.is_final is True
        assert [c["run_index"] for c in manifest_on_disk(mgr)["checkpoints"]] == [0]
        assert "filelock unavailable" in caplog.text
        assert "register" in caplog.text

    def test_zero_timeout_means_block_forever_and_works_uncontended(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "z"), lock_timeout_seconds=0)
        p = make_ckpt(m.checkpoint_dir, "c1")
        m.register(0, str(p), validation_loss=1.0)
        assert m._lock_timeout_seconds == 0.0
        assert len(manifest_on_disk(m)["checkpoints"]) == 1

    def test_lock_timeout_logs_and_degrades_to_unserialized_write(self, tmp_path, caplog):
        m = CheckpointManager(str(tmp_path / "t"), CheckpointPolicy(auto_prune=False),
                              lock_timeout_seconds=0.05)
        p = make_ckpt(m.checkpoint_dir, "c1")
        holder = FileLock(str(m._lock_path))
        with holder, caplog.at_level(logging.ERROR, logger=LOGGER):
            info = m.register(3, str(p), validation_loss=0.25)
        assert "lock acquisition timed out after 0.1s" in caplog.text
        assert "_locked_manifest_write(register)" in caplog.text
        # Best-effort contract: the write still happened, unserialized.
        assert info.run_index == 3
        assert [c["run_index"] for c in manifest_on_disk(m)["checkpoints"]] == [3]

    def test_lock_timeout_on_every_mutator_still_completes(self, tmp_path, caplog):
        m = CheckpointManager(
            str(tmp_path / "t2"),
            CheckpointPolicy(keep_best_n=0, keep_final=False, max_total=0, auto_prune=False),
            lock_timeout_seconds=0.05,
        )
        a = make_ckpt(m.checkpoint_dir, "a")
        b = make_ckpt(m.checkpoint_dir, "b")
        m.register(0, str(a), validation_loss=0.9)
        m.register(1, str(b), validation_loss=0.8)
        with FileLock(str(m._lock_path)), caplog.at_level(logging.ERROR, logger=LOGGER):
            assert m.protect_checkpoint(0) is True
            assert m.unprotect_checkpoint(0) is True
            assert m.cleanup_orphaned() == 0
            shutil.rmtree(b)
            assert m.cleanup_orphaned() == 1
            pruned = m.prune()
            assert [c.run_index for c in pruned] == [0]
            assert m.force_prune_to_size(0.0) == []
        for op in ("protect_checkpoint", "unprotect_checkpoint", "cleanup_orphaned",
                   "prune", "force_prune_to_size"):
            assert f"_locked_manifest_write({op})" in caplog.text


# ---------------------------------------------------------------------------
# Manifest save
# ---------------------------------------------------------------------------

class TestSaveManifestFailures:
    def test_fsync_unsupported_is_tolerated(self, mgr, monkeypatch, caplog):
        def no_fsync(fd):
            raise OSError("fsync unsupported")

        monkeypatch.setattr(ckpt_mod.os, "fsync", no_fsync)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert mgr._save_manifest() is True
        assert "fsync skipped" in caplog.text
        assert manifest_on_disk(mgr)["version"] == CheckpointManager.CURRENT_MANIFEST_VERSION

    def test_replace_failure_returns_false_and_removes_tmp(self, mgr, monkeypatch, caplog):
        fail_replace(monkeypatch)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            assert mgr._save_manifest() is False
        assert "Failed to save checkpoint manifest" in caplog.text
        assert not mgr._manifest_path.with_suffix(".json.tmp").exists()

    def test_tmp_cleanup_failure_is_swallowed(self, mgr, monkeypatch):
        fail_replace(monkeypatch)
        real_unlink = Path.unlink

        def deny(self, *a, **k):
            if self.name.endswith(".tmp"):
                raise PermissionError("tmp locked")
            return real_unlink(self, *a, **k)

        monkeypatch.setattr(Path, "unlink", deny)
        assert mgr._save_manifest() is False  # no exception escapes

    def test_failure_before_tmp_is_created_skips_cleanup(self, mgr, monkeypatch):
        def boom(_obj):
            raise TypeError("policy not serialisable")

        monkeypatch.setattr(ckpt_mod, "asdict", boom)
        assert mgr._save_manifest() is False
        assert not mgr._manifest_path.exists()

    def test_register_warns_when_save_returns_false(self, mgr, monkeypatch, caplog):
        fail_replace(monkeypatch)
        p = make_ckpt(mgr.checkpoint_dir, "c1")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            info = mgr.register(0, str(p), validation_loss=0.5)
        assert "Manifest save failed after registering checkpoint" in caplog.text
        # In-memory state still has it even though disk does not.
        assert mgr.list_checkpoints() == [info]
        assert not mgr._manifest_path.exists()

    def test_register_warns_when_save_raises(self, mgr, monkeypatch, caplog):
        def boom():
            raise RuntimeError("save exploded")

        monkeypatch.setattr(mgr, "_save_manifest", boom)
        p = make_ckpt(mgr.checkpoint_dir, "c1")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            mgr.register(0, str(p))
        assert "Manifest save error after registering checkpoint: save exploded" in caplog.text


# ---------------------------------------------------------------------------
# Size / lookup / scoring
# ---------------------------------------------------------------------------

class TestSizeAndLookup:
    def test_size_of_nested_directory_counts_only_files(self, mgr):
        p = mgr.checkpoint_dir / "nest"
        (p / "sub" / "deeper").mkdir(parents=True)
        (p / "a.bin").write_bytes(b"1" * 10)
        (p / "sub" / "b.bin").write_bytes(b"2" * 20)
        (p / "sub" / "deeper" / "c.bin").write_bytes(b"3" * 30)
        assert mgr._get_checkpoint_size(str(p)) == 60

    def test_size_of_single_file_and_missing_path(self, mgr):
        f = make_ckpt(mgr.checkpoint_dir, "model.pt", size=123, as_file=True)
        assert mgr._get_checkpoint_size(str(f)) == 123
        assert mgr._get_checkpoint_size(str(mgr.checkpoint_dir / "nope")) == 0

    def test_find_latest_for_run_id_picks_highest_run_index(self, mgr):
        for idx, rid in [(0, "A"), (2, "A"), (1, "A"), (5, "B")]:
            mgr.register(idx, str(make_ckpt(mgr.checkpoint_dir, f"c{idx}")), run_id=rid)
        found = mgr.find_latest_for_run_id("A")
        assert found is not None and found.run_index == 2 and found.run_id == "A"
        assert mgr.find_latest_for_run_id("B").run_index == 5
        assert mgr.find_latest_for_run_id("missing") is None

    def test_score_without_validation_loss_gets_only_flag_bonuses(self, tmp_path):
        m = CheckpointManager(
            str(tmp_path / "s"),
            CheckpointPolicy(keep_best_n=2, keep_final=True, keep_run_boundaries=True,
                             auto_prune=False),
        )
        none_loss = CheckpointInfo(run_index=0, path="x", validation_loss=None)
        m._checkpoints = [none_loss]
        assert m._score_checkpoint(none_loss) == 0.0
        none_loss.is_final = True
        none_loss.is_run_boundary = True
        assert m._score_checkpoint(none_loss) == 1500.0
        none_loss.protected = True
        assert m._score_checkpoint(none_loss) == float("inf")

    def test_score_loss_outside_top_n_is_zero_and_best_rank_is_scaled(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "s2"),
                              CheckpointPolicy(keep_best_n=2, keep_final=False, auto_prune=False))
        cps = [CheckpointInfo(run_index=i, path=f"p{i}", validation_loss=loss)
               for i, loss in enumerate([0.1, 0.2, 0.3])]
        m._checkpoints = cps
        assert [m._score_checkpoint(c) for c in cps] == [200.0, 100.0, 0.0]

    def test_best_n_with_negative_keep_clamps_to_empty(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "s3"),
                              CheckpointPolicy(keep_best_n=-4, auto_prune=False))
        m._checkpoints = [CheckpointInfo(run_index=0, path="p", validation_loss=0.1)]
        assert m._best_n_by_val_loss() == []

    def test_tied_losses_keep_exactly_n(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "s4"),
                              CheckpointPolicy(keep_best_n=1, keep_final=False, auto_prune=False))
        m._checkpoints = [CheckpointInfo(run_index=i, path=f"p{i}", validation_loss=0.5,
                                         timestamp=f"2026-01-0{i + 1}") for i in range(3)]
        prunable = m._get_prunable_checkpoints()
        assert sorted(c.run_index for c in prunable) == [1, 2]

    def test_run_boundary_checkpoints_are_never_prunable(self, tmp_path):
        m = CheckpointManager(
            str(tmp_path / "s5"),
            CheckpointPolicy(keep_best_n=0, keep_final=False, keep_run_boundaries=True,
                             auto_prune=False),
        )
        boundary = CheckpointInfo(run_index=0, path="b", is_run_boundary=True)
        plain = CheckpointInfo(run_index=1, path="p")
        m._checkpoints = [boundary, plain]
        assert m._get_prunable_checkpoints() == [plain]

    def test_stats_summary_includes_best_line(self, mgr):
        mgr.register(0, str(make_ckpt(mgr.checkpoint_dir, "a")), validation_loss=0.25)
        stats = mgr.get_stats()
        assert isinstance(stats, CheckpointStats)
        assert "Best: Run 0 (val_loss=0.2500)" in stats.summary()
        # keep_best_n=1: the lone checkpoint is the retained best, so nothing is prunable
        assert stats.total_count == 1 and stats.prunable_count == 0
        assert stats.best_checkpoint.run_index == 0 and stats.protected_count == 0

    def test_stats_summary_omits_best_without_loss(self):
        s = CheckpointStats(total_count=2, protected_count=1, prunable_count=1)
        assert "Best:" not in s.summary()
        assert s.summary().splitlines()[1] == "Protected: 1, Prunable: 1"

    def test_get_final_checkpoint_none_when_empty(self, mgr):
        assert mgr.get_final_checkpoint() is None
        assert mgr.get_best_checkpoint() is None
        assert mgr.get_stats() == CheckpointStats()


# ---------------------------------------------------------------------------
# Manifest load
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_old_version_warns_and_unknown_keys_dropped(self, tmp_path, caplog):
        d = tmp_path / "old"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "checkpoints": [{"run_index": 4, "path": "p", "from_the_future": True}],
        }))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            m = CheckpointManager(str(d))
        assert "version='0.0'" in caplog.text
        assert [c.run_index for c in m.list_checkpoints()] == [4]

    def test_corrupt_manifest_is_reset_with_warning(self, tmp_path, caplog):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "manifest.json").write_text("{not json")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            m = CheckpointManager(str(d))
        assert m.list_checkpoints() == []
        assert "Failed to load manifest" in caplog.text


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def seed(m: CheckpointManager, n: int, *, as_file: bool = False, losses=None):
    """Register ``n`` checkpoints (auto-prune must be off on the manager)."""
    paths = []
    for i in range(n):
        name = f"c{i}.pt" if as_file else f"c{i}"
        p = make_ckpt(m.checkpoint_dir, name, as_file=as_file)
        loss = None if losses is None else losses[i]
        m.register(i, str(p), validation_loss=loss)
        paths.append(p)
    return paths


class TestPrune:
    def test_dry_run_returns_plan_without_deleting(self, mgr, caplog):
        paths = seed(mgr, 3, losses=[0.9, 0.5, 0.7])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            plan = mgr.prune(dry_run=True)
        assert sorted(c.run_index for c in plan) == [0, 2]
        assert all(p.exists() for p in paths)
        assert len(mgr.list_checkpoints()) == 3
        assert "Dry run: would prune 2" in caplog.text

    def test_prune_nothing_to_do(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "n"), CheckpointPolicy(
            keep_best_n=5, keep_final=True, max_total=0, auto_prune=False))
        seed(m, 2, losses=[0.1, 0.2])
        assert m.prune() == []

    def test_prune_deletes_file_and_directory_checkpoints(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "mix"), CheckpointPolicy(
            keep_best_n=1, keep_final=False, max_total=0, auto_prune=False))
        d = make_ckpt(m.checkpoint_dir, "dir_ckpt")
        f = make_ckpt(m.checkpoint_dir, "file.pt", as_file=True)
        keep = make_ckpt(m.checkpoint_dir, "keep")
        m.register(0, str(d), validation_loss=0.9)
        m.register(1, str(f), validation_loss=0.8)
        m.register(2, str(keep), validation_loss=0.1)
        pruned = m.prune()
        assert sorted(c.run_index for c in pruned) == [0, 1]
        assert not d.exists() and not f.exists() and keep.exists()
        assert [c["run_index"] for c in manifest_on_disk(m)["checkpoints"]] == [2]

    def test_prune_tolerates_checkpoint_already_gone_from_disk(self, mgr):
        paths = seed(mgr, 2, losses=[0.9, 0.1])
        shutil.rmtree(paths[0])
        pruned = mgr.prune()
        assert [c.run_index for c in pruned] == [0]
        assert [c.run_index for c in mgr.list_checkpoints()] == [1]

    def test_partial_directory_delete_drops_entry_and_finishes_teardown(
        self, mgr, monkeypatch, caplog
    ):
        paths = seed(mgr, 2, losses=[0.9, 0.1])
        real = shutil.rmtree

        def flaky(path, ignore_errors=False, **kw):
            if not ignore_errors:
                raise OSError("file in use")
            return real(path, ignore_errors=True, **kw)

        monkeypatch.setattr(ckpt_mod.shutil, "rmtree", flaky)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            pruned = mgr.prune()
        assert [c.run_index for c in pruned] == [0]
        assert "Failed to prune checkpoint" in caplog.text
        assert not paths[0].exists()  # best-effort retry removed it
        assert [c.run_index for c in mgr.list_checkpoints()] == [1]
        assert [c["run_index"] for c in manifest_on_disk(mgr)["checkpoints"]] == [1]

    def test_partial_file_delete_retries_with_missing_ok(self, tmp_path, monkeypatch):
        m = CheckpointManager(str(tmp_path / "pf"), CheckpointPolicy(
            keep_best_n=1, keep_final=False, max_total=0, auto_prune=False))
        paths = seed(m, 2, as_file=True, losses=[0.9, 0.1])
        real_unlink = Path.unlink
        calls = {"n": 0}

        def flaky(self, missing_ok=False):
            if self.name == "c0.pt":
                calls["n"] += 1
                if calls["n"] == 1:
                    raise PermissionError("antivirus holds it")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", flaky)
        pruned = m.prune()
        assert [c.run_index for c in pruned] == [0]
        assert not paths[0].exists()
        assert calls["n"] == 2

    def test_cleanup_failure_still_drops_manifest_entry(self, mgr, monkeypatch, caplog):
        seed(mgr, 2, losses=[0.9, 0.1])

        def always(path, ignore_errors=False, **kw):
            raise OSError("handle leak")

        monkeypatch.setattr(ckpt_mod.shutil, "rmtree", always)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            pruned = mgr.prune()
        assert [c.run_index for c in pruned] == [0]
        assert "also failed: handle leak" in caplog.text
        assert [c.run_index for c in mgr.list_checkpoints()] == [1]

    def test_save_failure_after_prune_recovers_orphans_once_disk_returns(
        self, mgr, monkeypatch, caplog
    ):
        paths = seed(mgr, 3, losses=[0.9, 0.5, 0.1])
        # Only the first save (right after deleting) fails; recovery succeeds.
        fail_replace(monkeypatch, times=1)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            pruned = mgr.prune()
        assert len(pruned) == 2
        assert "attempting orphan-entry recovery" in caplog.text
        assert "Orphan recovery removed 2 stale manifest entries" in caplog.text
        assert not paths[0].exists() and not paths[1].exists()
        assert [c["run_index"] for c in manifest_on_disk(mgr)["checkpoints"]] == [2]

    def test_save_failure_persisting_through_recovery_is_reported(
        self, mgr, monkeypatch, caplog
    ):
        seed(mgr, 2, losses=[0.9, 0.1])
        fail_replace(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            pruned = mgr.prune()
        assert len(pruned) == 1
        assert "Orphan-recovery manifest save also failed" in caplog.text
        assert "cleanup_orphaned()" in caplog.text

    def test_save_raising_during_prune_is_contained(self, mgr, monkeypatch, caplog):
        seed(mgr, 2, losses=[0.9, 0.1])

        def boom():
            raise RuntimeError("serializer crashed")

        monkeypatch.setattr(mgr, "_save_manifest", boom)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            pruned = mgr.prune()
        assert len(pruned) == 1
        assert "Manifest save error after pruning: serializer crashed" in caplog.text
        assert "Orphan-entry recovery after prune save failure raised" in caplog.text


# ---------------------------------------------------------------------------
# protect / unprotect / cleanup_orphaned
# ---------------------------------------------------------------------------

class TestProtectUnprotect:
    def test_protect_unknown_run_returns_false(self, mgr):
        assert mgr.protect_checkpoint(99) is False
        assert mgr.unprotect_checkpoint(99) is False

    def test_protect_roundtrip_persists_flag(self, mgr):
        seed(mgr, 1)
        assert mgr.protect_checkpoint(0) is True
        assert manifest_on_disk(mgr)["checkpoints"][0]["protected"] is True
        assert mgr.unprotect_checkpoint(0) is True
        assert manifest_on_disk(mgr)["checkpoints"][0]["protected"] is False

    @pytest.mark.parametrize("op,attr", [("protect_checkpoint", "Manifest save failed after protecting"),
                                         ("unprotect_checkpoint", "Manifest save failed after unprotecting")])
    def test_save_returning_false_warns_but_still_flips_flag(
        self, mgr, monkeypatch, caplog, op, attr
    ):
        seed(mgr, 1)
        monkeypatch.setattr(mgr, "_save_manifest", lambda: False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert getattr(mgr, op)(0) is True
        assert attr in caplog.text

    @pytest.mark.parametrize("op,msg", [("protect_checkpoint", "save error after protecting checkpoint: kaput"),
                                        ("unprotect_checkpoint", "save error after unprotecting checkpoint: kaput")])
    def test_save_raising_is_contained(self, mgr, monkeypatch, caplog, op, msg):
        seed(mgr, 1)

        def boom():
            raise RuntimeError("kaput")

        monkeypatch.setattr(mgr, "_save_manifest", boom)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert getattr(mgr, op)(0) is True
        assert msg in caplog.text


class TestCleanupOrphaned:
    def test_removes_missing_paths_and_persists(self, mgr, caplog):
        paths = seed(mgr, 3)
        shutil.rmtree(paths[1])
        with caplog.at_level(logging.INFO, logger=LOGGER):
            assert mgr.cleanup_orphaned() == 1
        assert [c["run_index"] for c in manifest_on_disk(mgr)["checkpoints"]] == [0, 2]
        assert "Removed orphaned manifest entry" in caplog.text

    def test_nothing_orphaned_returns_zero_without_rewriting(self, mgr):
        seed(mgr, 1)
        before = mgr._manifest_path.read_text()
        assert mgr.cleanup_orphaned() == 0
        assert mgr._manifest_path.read_text() == before

    def test_save_failure_is_warned(self, mgr, monkeypatch, caplog):
        paths = seed(mgr, 2)
        shutil.rmtree(paths[0])
        monkeypatch.setattr(mgr, "_save_manifest", lambda: False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert mgr.cleanup_orphaned() == 1
        assert "Manifest save failed after cleaning orphaned entries" in caplog.text

    def test_save_raising_is_warned(self, mgr, monkeypatch, caplog):
        paths = seed(mgr, 2)
        shutil.rmtree(paths[0])

        def boom():
            raise RuntimeError("nope")

        monkeypatch.setattr(mgr, "_save_manifest", boom)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert mgr.cleanup_orphaned() == 1
        assert "Manifest save error after cleaning orphaned entries: nope" in caplog.text


# ---------------------------------------------------------------------------
# force_prune_to_size
# ---------------------------------------------------------------------------

GB = 1024**3


def write_manifest(d: Path, entries: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"version": "1.0", "checkpoints": entries}))


def set_sizes(m: CheckpointManager, size: int) -> None:
    """Pretend every checkpoint is ``size`` bytes (persisted: prune re-reads disk)."""
    for cp in m._checkpoints:
        cp.size_bytes = size
    assert m._save_manifest() is True


class TestForcePrune:
    def _mgr(self, tmp_path, **policy):
        base = {"keep_best_n": 0, "keep_final": False, "keep_run_boundaries": False,
                "max_total": 0, "auto_prune": False}
        base.update(policy)
        return CheckpointManager(str(tmp_path / "fp"), CheckpointPolicy(**base))

    def test_prunes_lowest_scored_until_under_limit_files_and_dirs(self, tmp_path):
        m = self._mgr(tmp_path)
        d = make_ckpt(m.checkpoint_dir, "d")
        f = make_ckpt(m.checkpoint_dir, "f.pt", as_file=True)
        m.register(0, str(d))
        m.register(1, str(f))
        set_sizes(m, GB)
        pruned = m.force_prune_to_size(1.0)
        assert [c.run_index for c in pruned] == [0]  # ties resolve in manifest order
        assert sum(c.size_bytes for c in m.list_checkpoints()) <= GB
        assert not d.exists() and f.exists()

    def test_file_checkpoint_victim_is_unlinked(self, tmp_path):
        m = self._mgr(tmp_path)
        f = make_ckpt(m.checkpoint_dir, "only.pt", as_file=True)
        m.register(0, str(f))
        set_sizes(m, GB)
        pruned = m.force_prune_to_size(0.0)
        assert [c.run_index for c in pruned] == [0]
        assert not f.exists()

    def test_victim_already_missing_from_disk_is_still_dropped(self, tmp_path):
        m = self._mgr(tmp_path)
        m.register(0, str(make_ckpt(m.checkpoint_dir, "gone")))
        set_sizes(m, GB)
        shutil.rmtree(m.checkpoint_dir / "gone")
        assert [c.run_index for c in m.force_prune_to_size(0.0)] == [0]
        assert m.list_checkpoints() == []

    def test_all_protected_by_policy_stops_with_actionable_warning(self, tmp_path, caplog):
        m = self._mgr(tmp_path, keep_final=True)
        m.register(0, str(make_ckpt(m.checkpoint_dir, "x")))
        set_sizes(m, 4 * GB)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert m.force_prune_to_size(1.0) == []
        assert "Cannot prune further: all 1 remaining checkpoints" in caplog.text
        assert len(m.list_checkpoints()) == 1

    def test_already_under_limit_is_noop(self, tmp_path):
        m = self._mgr(tmp_path)
        m.register(0, str(make_ckpt(m.checkpoint_dir, "x")))
        assert m.force_prune_to_size(10.0) == []

    def test_delete_failure_drops_entry_and_stops_loop(self, tmp_path, monkeypatch, caplog):
        m = self._mgr(tmp_path)
        a = make_ckpt(m.checkpoint_dir, "a")
        b = make_ckpt(m.checkpoint_dir, "b")
        m.register(0, str(a))
        m.register(1, str(b))
        set_sizes(m, GB)
        real = shutil.rmtree

        def flaky(path, ignore_errors=False, **kw):
            if not ignore_errors:
                raise OSError("sharing violation")
            return real(path, ignore_errors=True, **kw)

        monkeypatch.setattr(ckpt_mod.shutil, "rmtree", flaky)
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            pruned = m.force_prune_to_size(0.0)
        # Loop breaks after the first failed victim, entry dropped, dir finished off.
        assert len(pruned) == 1
        assert "Failed to force-prune" in caplog.text
        victim_dir = Path(pruned[0].path)
        assert not victim_dir.exists()
        assert len(m.list_checkpoints()) == 1

    def test_delete_failure_on_file_victim_uses_missing_ok_retry(self, tmp_path, monkeypatch):
        m = self._mgr(tmp_path)
        f = make_ckpt(m.checkpoint_dir, "v.pt", as_file=True)
        m.register(0, str(f))
        set_sizes(m, GB)
        real_unlink = Path.unlink
        calls = {"n": 0}

        def flaky(self, missing_ok=False):
            if self.name == "v.pt":
                calls["n"] += 1
                if calls["n"] == 1:
                    raise PermissionError("locked")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", flaky)
        pruned = m.force_prune_to_size(0.0)
        assert [c.run_index for c in pruned] == [0]
        assert not f.exists() and calls["n"] == 2

    def test_delete_and_cleanup_both_failing_is_warned(self, tmp_path, monkeypatch, caplog):
        m = self._mgr(tmp_path)
        m.register(0, str(make_ckpt(m.checkpoint_dir, "a")))
        set_sizes(m, GB)

        def always(path, ignore_errors=False, **kw):
            raise OSError("wedged")

        monkeypatch.setattr(ckpt_mod.shutil, "rmtree", always)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            pruned = m.force_prune_to_size(0.0)
        assert [c.run_index for c in pruned] == [0]
        assert "also failed: wedged" in caplog.text

    def test_iteration_ceiling_aborts_with_diagnostics(self, tmp_path, caplog):
        d = tmp_path / "fp"
        write_manifest(d, [
            CheckpointInfo(run_index=i, path=str(d / f"missing{i}"), size_bytes=GB,
                           timestamp=f"2026-01-01T00:00:{i:02d}").to_dict()
            for i in range(105)
        ])
        m = CheckpointManager(str(d), CheckpointPolicy(
            keep_best_n=0, keep_final=False, max_total=0, auto_prune=False))
        with caplog.at_level(logging.ERROR, logger=LOGGER):
            pruned = m.force_prune_to_size(0.0)
        assert len(pruned) == 100
        assert len(m.list_checkpoints()) == 5
        assert "hit 100 iteration limit" in caplog.text
        assert "checkpoints_remaining=5" in caplog.text
        assert len(manifest_on_disk(m)["checkpoints"]) == 5

    def test_save_failure_after_force_prune_warns(self, tmp_path, monkeypatch, caplog):
        m = self._mgr(tmp_path)
        m.register(0, str(make_ckpt(m.checkpoint_dir, "a")))
        set_sizes(m, GB)
        monkeypatch.setattr(m, "_save_manifest", lambda: False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            m.force_prune_to_size(0.0)
        assert "Manifest save failed after force prune" in caplog.text

    def test_save_raising_after_force_prune_warns(self, tmp_path, monkeypatch, caplog):
        m = self._mgr(tmp_path)
        m.register(0, str(make_ckpt(m.checkpoint_dir, "a")))
        set_sizes(m, GB)

        def boom():
            raise RuntimeError("bad")

        monkeypatch.setattr(m, "_save_manifest", boom)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            m.force_prune_to_size(0.0)
        assert "Manifest save error after force prune: bad" in caplog.text


# ---------------------------------------------------------------------------
# Prune plan details, partial-delete corner cases
# ---------------------------------------------------------------------------

class TestPrunePlan:
    def test_protected_checkpoint_is_never_in_plan_or_prunable(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "pl"), CheckpointPolicy(
            keep_best_n=0, keep_final=False, max_total=0, auto_prune=False))
        keep = CheckpointInfo(run_index=0, path="k", protected=True)
        drop = CheckpointInfo(run_index=1, path="d")
        m._checkpoints = [keep, drop]
        assert m._get_prunable_checkpoints() == [drop]
        assert m._compute_prune_plan() == [drop]

    def test_max_total_caps_kept_checkpoints_by_score(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "mt"), CheckpointPolicy(
            keep_best_n=5, keep_final=False, max_total=2, auto_prune=False))
        cps = [CheckpointInfo(run_index=i, path=f"p{i}", validation_loss=0.1 * (i + 1))
               for i in range(4)]
        m._checkpoints = list(cps)
        plan = m._compute_prune_plan()
        # Scores: run0 500, run1 400, run2 300, run3 200 -> only the 2 best survive.
        assert sorted(c.run_index for c in plan) == [2, 3]

    def test_protected_entries_occupy_max_total_slots(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "mt2"), CheckpointPolicy(
            keep_best_n=5, keep_final=False, max_total=1, auto_prune=False))
        prot = CheckpointInfo(run_index=0, path="p", protected=True)
        a = CheckpointInfo(run_index=1, path="a", validation_loss=0.1)
        b = CheckpointInfo(run_index=2, path="b", validation_loss=0.2)
        m._checkpoints = [prot, a, b]
        # The protected entry is kept first (infinite score) and fills the single slot,
        # so both unprotected entries are pruned.
        assert m._compute_prune_plan() == [a, b]

    def test_protect_picks_matching_run_among_several(self, mgr):
        seed(mgr, 3)
        assert mgr.protect_checkpoint(2) is True
        flags = {c.run_index: c.protected for c in mgr.list_checkpoints()}
        assert flags == {0: False, 1: False, 2: True}
        assert mgr.unprotect_checkpoint(2) is True
        assert not any(c.protected for c in mgr.list_checkpoints())

    def test_delete_that_removes_dir_then_raises_skips_retry(self, mgr, monkeypatch):
        paths = seed(mgr, 2, losses=[0.9, 0.1])
        real = shutil.rmtree

        def remove_then_fail(path, ignore_errors=False, **kw):
            real(path, ignore_errors=True)
            raise OSError("late failure")

        monkeypatch.setattr(ckpt_mod.shutil, "rmtree", remove_then_fail)
        pruned = mgr.prune()
        assert [c.run_index for c in pruned] == [0]
        assert not paths[0].exists()
        assert [c.run_index for c in mgr.list_checkpoints()] == [1]

    def test_force_delete_that_removes_dir_then_raises_skips_retry(self, tmp_path, monkeypatch):
        m = CheckpointManager(str(tmp_path / "fd"), CheckpointPolicy(
            keep_best_n=0, keep_final=False, max_total=0, auto_prune=False))
        p = make_ckpt(m.checkpoint_dir, "a")
        m.register(0, str(p))
        set_sizes(m, GB)
        real = shutil.rmtree

        def remove_then_fail(path, ignore_errors=False, **kw):
            real(path, ignore_errors=True)
            raise OSError("late failure")

        monkeypatch.setattr(ckpt_mod.shutil, "rmtree", remove_then_fail)
        pruned = m.force_prune_to_size(0.0)
        assert [c.run_index for c in pruned] == [0]
        assert not p.exists() and m.list_checkpoints() == []
