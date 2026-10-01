"""Saving into a directory replaces only what the save writes.

Pre-fix, Trainer.save / export_lora / SLAOMerger.save promoted their staged
``<path>.partial`` by replacing the WHOLE target directory. ``backprop train
--output X`` uses X both as the Trainer's ``output_dir`` (run_history.json,
checkpoint-N/) and as the save path, so every CLI training run deleted its
own run history, its intermediate checkpoints and any unrelated files in X.

These tests pin the fixed contract: the promote replaces same-named entries
only, keeps its crash safety (journal + rollback), and leaves the rest of the
directory alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backpropagate.checkpoints import (
    RunHistoryManager,
    promote_partial_dir,
    recover_interrupted_promote,
)


def _seed_output_dir(out: Path) -> None:
    """An output dir as a previous run + the operator left it."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_history.json").write_text('{"runs": []}', encoding="utf-8")
    (out / "checkpoint-5").mkdir()
    (out / "checkpoint-5" / "trainer_state.json").write_text("{}", encoding="utf-8")
    (out / "my-notes.txt").write_text("keep me", encoding="utf-8")


def _assert_seed_intact(out: Path) -> None:
    assert (out / "run_history.json").exists(), "run_history.json was deleted"
    assert (out / "checkpoint-5" / "trainer_state.json").exists(), (
        "intermediate checkpoint-5/ was deleted"
    )
    assert (out / "my-notes.txt").read_text(encoding="utf-8") == "keep me", (
        "an unrelated user file was deleted"
    )


def _no_leftovers(out: Path) -> None:
    for suffix in (".partial", ".backup", ".backup.json", ".backup.json.tmp"):
        leftover = out.with_name(out.name + suffix)
        assert not leftover.exists(), f"{leftover} left behind"


def _make_trainer(output_dir: Path, weights: bytes = b"NEW-WEIGHTS", trainer_cls=None):
    from backpropagate.trainer import Trainer

    trainer_cls = trainer_cls or Trainer
    with patch("torch.cuda.is_available", return_value=False):
        trainer = trainer_cls(output_dir=str(output_dir), use_unsloth=False)
    trainer._model = MagicMock()
    trainer._tokenizer = MagicMock()
    trainer._is_loaded = True
    trainer._has_trained = True
    _set_weights(trainer, weights)
    return trainer


def _set_weights(trainer, weights: bytes) -> None:
    def write_model(path, *args, **kwargs):
        (Path(path) / "adapter_model.safetensors").write_bytes(weights)
        (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")

    def write_tok(path, *args, **kwargs):
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    trainer._model.save_pretrained.side_effect = write_model
    trainer._tokenizer.save_pretrained.side_effect = write_tok


# =============================================================================
# Trainer.save
# =============================================================================


class TestTrainerSavePreservesOutputDir:
    def test_save_into_populated_dir_keeps_other_files(self, temp_dir):
        out = temp_dir / "output"
        _seed_output_dir(out)
        (out / "adapter_model.safetensors").write_bytes(b"OLD-WEIGHTS")
        trainer = _make_trainer(out)

        trainer.save(str(out))

        _assert_seed_intact(out)
        assert (out / "adapter_model.safetensors").read_bytes() == b"NEW-WEIGHTS"
        assert (out / "adapter_config.json").exists()
        assert (out / "tokenizer.json").exists()
        _no_leftovers(out)

    def test_saving_twice_replaces_model_files(self, temp_dir):
        out = temp_dir / "output"
        _seed_output_dir(out)
        trainer = _make_trainer(out, weights=b"FIRST")
        trainer.save(str(out))
        assert (out / "adapter_model.safetensors").read_bytes() == b"FIRST"

        _set_weights(trainer, b"SECOND")
        trainer.save(str(out))

        assert (out / "adapter_model.safetensors").read_bytes() == b"SECOND"
        _assert_seed_intact(out)
        _no_leftovers(out)

    def test_stale_run_id_does_not_survive_a_save_without_one(self, temp_dir):
        out = temp_dir / "output"
        _seed_output_dir(out)
        trainer = _make_trainer(out, weights=b"FIRST")
        trainer.save(str(out), run_id="run-old")
        assert (out / "run_id").read_text(encoding="utf-8") == "run-old"

        _set_weights(trainer, b"SECOND")
        trainer.save(str(out))

        assert not (out / "run_id").exists(), (
            "a run_id from an earlier save must not label the new weights"
        )
        assert (out / "adapter_model.safetensors").read_bytes() == b"SECOND"
        _assert_seed_intact(out)
        _no_leftovers(out)

    def test_save_into_fresh_dir_still_works(self, temp_dir):
        out = temp_dir / "fresh"
        trainer = _make_trainer(out)
        trainer.save(str(out))
        assert (out / "adapter_model.safetensors").read_bytes() == b"NEW-WEIGHTS"
        _no_leftovers(out)

    def test_failed_promote_restores_prior_files_and_keeps_others(self, temp_dir):
        """A move failing on the SECOND entry rolls back the first one too."""
        from backpropagate.exceptions import CheckpointError

        out = temp_dir / "output"
        _seed_output_dir(out)
        (out / "adapter_config.json").write_text("OLD-CONFIG", encoding="utf-8")
        (out / "adapter_model.safetensors").write_bytes(b"OLD-WEIGHTS")
        trainer = _make_trainer(out)

        real_move = shutil.move
        calls = {"n": 0}

        def flaky_move(src, dst, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("[Errno 28] No space left on device")
            return real_move(src, dst, *args, **kwargs)

        with patch.object(shutil, "move", side_effect=flaky_move), \
             pytest.raises(CheckpointError):
            trainer.save(str(out))

        # sorted order: adapter_config.json moved in, then the weights failed.
        assert (out / "adapter_config.json").read_text(encoding="utf-8") == "OLD-CONFIG"
        assert (out / "adapter_model.safetensors").read_bytes() == b"OLD-WEIGHTS"
        assert not (out / "tokenizer.json").exists(), (
            "a new entry with no prior must be removed on rollback"
        )
        _assert_seed_intact(out)
        _no_leftovers(out)


# =============================================================================
# promote_partial_dir / recover_interrupted_promote (crash windows)
# =============================================================================


class TestPromoteCrashRecovery:
    def _staged(self, temp_dir: Path):
        target = temp_dir / "out"
        _seed_output_dir(target)
        (target / "a.bin").write_bytes(b"OLD-A")
        (target / "b.bin").write_bytes(b"OLD-B")
        partial = temp_dir / "out.partial"
        partial.mkdir()
        (partial / "a.bin").write_bytes(b"NEW-A")
        (partial / "b.bin").write_bytes(b"NEW-B")
        (partial / "c.bin").write_bytes(b"NEW-C")
        return target, partial

    def test_crash_mid_promote_rolls_back_to_complete_prior_save(self, temp_dir):
        """Simulate the on-disk state of a process killed after promoting
        ``a.bin`` but before ``b.bin``: journal present, old a+b in backup,
        new a in place. Recovery must restore the prior save in full, never a
        mix of old and new files."""
        target, partial = self._staged(temp_dir)
        backup = temp_dir / "out.backup"
        journal = temp_dir / "out.backup.json"
        backup.mkdir()
        journal.write_text(json.dumps({"entries": [
            {"name": "a.bin", "had_prior": True},
            {"name": "b.bin", "had_prior": True},
            {"name": "c.bin", "had_prior": False},
        ]}), encoding="utf-8")
        (target / "a.bin").rename(backup / "a.bin")
        shutil.move(str(partial / "a.bin"), str(target / "a.bin"))
        (target / "b.bin").rename(backup / "b.bin")

        recover_interrupted_promote(target)

        assert (target / "a.bin").read_bytes() == b"OLD-A"
        assert (target / "b.bin").read_bytes() == b"OLD-B"
        assert not (target / "c.bin").exists()
        assert not backup.exists()
        assert not journal.exists()
        _assert_seed_intact(target)

    def test_committed_promote_with_leftover_backup_is_cleaned(self, temp_dir):
        """Crash after the journal was deleted (commit) but before the backup
        was: the new save stands, the backup is discarded."""
        target, _partial = self._staged(temp_dir)
        backup = temp_dir / "out.backup"
        backup.mkdir()
        (backup / "a.bin").write_bytes(b"OLD-A-STALE")

        recover_interrupted_promote(target)

        assert not backup.exists()
        assert (target / "a.bin").read_bytes() == b"OLD-A"
        _assert_seed_intact(target)

    def test_legacy_whole_dir_backup_is_recovered(self, temp_dir):
        """A pre-fix save that crashed left the whole prior dir at .backup
        and nothing at the target: recovery moves it back."""
        target = temp_dir / "out"
        backup = temp_dir / "out.backup"
        _seed_output_dir(backup)

        recover_interrupted_promote(target)

        assert not backup.exists()
        _assert_seed_intact(target)

    def test_promote_refuses_unresolved_leftovers(self, temp_dir):
        target, partial = self._staged(temp_dir)
        (temp_dir / "out.backup.json").write_text("not json", encoding="utf-8")
        (temp_dir / "out.backup").mkdir()

        recover_interrupted_promote(target)  # logs, does not raise
        with pytest.raises(FileExistsError):
            promote_partial_dir(partial, target)
        assert (target / "a.bin").read_bytes() == b"OLD-A"

    def test_promote_onto_a_file_is_refused(self, temp_dir):
        target = temp_dir / "out"
        target.write_text("a file", encoding="utf-8")
        partial = temp_dir / "out.partial"
        partial.mkdir()
        (partial / "a.bin").write_bytes(b"NEW")
        with pytest.raises(NotADirectoryError):
            promote_partial_dir(partial, target)
        assert target.read_text(encoding="utf-8") == "a file"


# =============================================================================
# export_lora / SLAOMerger.save — same promote, same contract
# =============================================================================


class TestOtherSavePathsPreserveOutputDir:
    def test_export_lora_keeps_other_files(self, temp_dir):
        from backpropagate.export import export_lora

        src = temp_dir / "adapter"
        src.mkdir()
        (src / "adapter_model.safetensors").write_bytes(b"NEW-WEIGHTS")
        (src / "adapter_config.json").write_text("{}", encoding="utf-8")

        out = temp_dir / "export"
        _seed_output_dir(out)
        (out / "adapter_model.safetensors").write_bytes(b"OLD-WEIGHTS")

        export_lora(str(src), out, emit_model_card=False)

        assert (out / "adapter_model.safetensors").read_bytes() == b"NEW-WEIGHTS"
        _assert_seed_intact(out)
        _no_leftovers(out)

    def test_slao_save_keeps_other_files(self, temp_dir):
        torch = pytest.importorskip("torch")
        from backpropagate.slao import SLAOMerger

        merger = SLAOMerger()
        merger.initialize({
            "layer.lora_A.weight": torch.randn(4, 8),
            "layer.lora_B.weight": torch.randn(8, 4),
        })
        out = temp_dir / "slao"
        _seed_output_dir(out)

        merger.save(str(out))
        merger.save(str(out))

        assert (out / "merge_history.json").exists()
        assert (out / "merged_lora.pt").exists()
        _assert_seed_intact(out)
        _no_leftovers(out)


# =============================================================================
# CLI: backprop train --output X keeps X's run history
# =============================================================================


class TestCmdTrainKeepsRunHistory:
    def test_train_then_list_runs_shows_the_run(self, temp_dir, capsys):
        """The reported bug end to end, minus the GPU: training records the
        run in <output>/run_history.json and writes checkpoint-N/, then
        cmd_train saves into the same <output>. list-runs must still see the
        run afterwards."""
        from backpropagate.cli import cmd_list_runs, cmd_train
        from backpropagate.trainer import Trainer as RealTrainer

        out = temp_dir / "output"
        out.mkdir()
        (out / "my-notes.txt").write_text("keep me", encoding="utf-8")

        def fake_train(trainer, *args, **kwargs):
            history = RunHistoryManager(str(trainer.output_dir))
            history.record_run_started(run_id="run-1", model_name="test-model")
            (Path(trainer.output_dir) / "checkpoint-5").mkdir(parents=True)
            history.record_run_completed(run_id="run-1", final_loss=0.5, steps=10)
            result = MagicMock()
            result.final_loss = 0.5
            result.duration_seconds = 1.0
            result.run_id = "run-1"
            return result

        def trainer_factory(**kwargs):
            trainer = _make_trainer(Path(kwargs["output_dir"]), trainer_cls=RealTrainer)
            trainer.train = lambda *a, **kw: fake_train(trainer, *a, **kw)
            return trainer

        with patch("backpropagate.trainer.Trainer", side_effect=trainer_factory):
            rc = cmd_train(argparse.Namespace(
                data="test_data.jsonl",
                model="test-model",
                steps=10,
                samples=None,
                batch_size="2",
                lr=2e-4,
                lora_r=16,
                output=str(out),
                no_unsloth=True,
                verbose=False,
            ))
        assert rc == 0
        assert (out / "adapter_model.safetensors").read_bytes() == b"NEW-WEIGHTS"
        assert (out / "run_history.json").exists()
        assert (out / "checkpoint-5").is_dir()
        assert (out / "my-notes.txt").read_text(encoding="utf-8") == "keep me"
        _no_leftovers(out)

        capsys.readouterr()
        rc = cmd_list_runs(argparse.Namespace(
            output=str(out), status=None, limit=None, json=True,
        ))
        assert rc == 0
        runs = json.loads(capsys.readouterr().out)
        assert [r["run_id"] for r in runs] == ["run-1"]
        assert runs[0]["status"] == "completed"
