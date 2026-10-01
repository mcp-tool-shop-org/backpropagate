"""Resume behaviour of ``MultiRunTrainer.run()`` against real on-disk state.

Session 1 really runs (real adapter, SLAO merger, checkpoints, run history);
session 2 is a brand-new trainer + model resuming from the same directory.
Mocked at the true boundaries only: ``trl.SFTTrainer`` (``make_fake_sft``), the
inner ``Trainer`` model load (``FakeInnerTrainer`` around a real PEFT model)
and ``get_gpu_status``.
"""

from __future__ import annotations

import logging
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate.checkpoints import RunHistoryManager
from backpropagate.exceptions import BackpropagateError, CheckpointError, InvalidSettingError
from backpropagate.multi_run import MergeMode, MultiRunTrainer
from tests.test_multi_run_cov_support import (
    FakeInnerTrainer,
    _cfg,
    build_env,
    fill_adapter,
    install_fake_sft,
    make_fake_sft,
    text_dataset,
)

MULTI_RUN_LOGGER = "backpropagate.multi_run"


@pytest.fixture
def env(monkeypatch, tmp_path):
    return build_env(monkeypatch, tmp_path)


def _ok(sft, run):
    fill_adapter(sft.model, run=run)
    return type("R", (), {"training_loss": 1.0})()


# =============================================================================
# Resume
# =============================================================================


def _crashed_session(env, *, num_runs=3):
    """Session 1: runs 1 and 2 complete and checkpoint, run 3 diverges (NaN)."""
    mrt = env.build(
        [lambda s: _ok(s, 1), lambda s: _ok(s, 2), _nan_run], num_runs=num_runs
    )
    with pytest.raises(BackpropagateError):
        mrt.run(text_dataset(60))
    return mrt


def _nan_run(sft):
    fill_adapter(sft.model, run=3, b_value=float("nan"))
    return type("R", (), {"training_loss": 1.0})()


class TestRunResume:
    def test_explicit_resume_continues_from_the_last_checkpoint_with_slao_state(
        self, env, caplog
    ):
        first = _crashed_session(env)
        run_id = first._run_id
        # Run 3 diverged AFTER run 2's accumulator was persisted, so the trusted
        # accumulator is the one on disk (the in-memory one holds the NaN merge).
        acc_before = torch.load(
            env.tmp_path / "run_002" / "slao" / "merged_lora.pt", weights_only=True
        )
        assert all(torch.isfinite(v).all() for v in acc_before.values())

        # Session 2: brand-new model + trainer, same checkpoint dir.
        env.inner = FakeInnerTrainer()
        fake2 = make_fake_sft()
        install_fake_sft(env.monkeypatch, fake2)
        second = MultiRunTrainer(
            model="tiny-test", config=_cfg(env.tmp_path, num_runs=3), resume_from=run_id
        )

        with caplog.at_level(logging.INFO, logger=MULTI_RUN_LOGGER):
            result = second.run(text_dataset(60))

        # same correlation token; only run 3 executed in this session
        assert result.run_id == run_id
        assert [r.run_index for r in result.runs] == [3]
        assert any(f"run_resumed run_id={run_id}" in r.getMessage() for r in caplog.records)
        assert second._resume_start_run_idx == 3
        assert second._resume_slao_state_restored is True
        # the persisted accumulator was rehydrated from disk (run 2's merge)
        assert second._slao_merger.run_index == 3  # continued from run_index 2
        # Run 3 started from the SLAO init built from that accumulator:
        # B = accumulator B (from session 1), A orthogonal.
        start = fake2.created[0].start_params
        for k, v in start.items():
            if ".lora_B." in k:
                assert torch.allclose(v, acc_before[k], atol=1e-6), k
            elif ".lora_A." in k:
                assert torch.allclose(v @ v.T, torch.eye(v.shape[0]), atol=1e-5), k
        # and the merge of run 3 continues the EMA at run_index 3. The new fake's
        # first train() call fills B with 1.0 (its own call counter restarts).
        lam3 = 1 / math.sqrt(3)
        b_acc = acc_before[next(k for k in acc_before if ".lora_B." in k)][0, 0].item()
        merged = second._slao_merger.get_merged_lora()
        b_key = next(k for k in merged if ".lora_B." in k)
        assert merged[b_key][0, 0].item() == pytest.approx(
            b_acc + lam3 * (1.0 - b_acc), abs=1e-5
        )
        entry = RunHistoryManager(str(env.tmp_path)).get_run(run_id)
        assert entry["status"] == "completed"

    def test_resume_aborts_when_slao_state_restored_but_checkpoint_missing(self, env):
        first = _crashed_session(env)
        run_id = first._run_id
        # break the paired LoRA checkpoint but keep the SLAO accumulator
        for f in (env.tmp_path / "run_002" / "lora").iterdir():
            if f.name.startswith("adapter_model"):
                f.unlink()

        env.inner = FakeInnerTrainer()
        install_fake_sft(env.monkeypatch, make_fake_sft())
        second = MultiRunTrainer(
            model="tiny-test", config=_cfg(env.tmp_path, num_runs=3), resume_from=run_id
        )

        with pytest.raises(CheckpointError) as exc_info:
            second.run(text_dataset(60))

        assert exc_info.value.operation == "load"
        assert "inconsistent" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    def test_resume_without_slao_state_warns_and_continues(self, env, caplog):
        first = _crashed_session(env)
        run_id = first._run_id
        import shutil

        shutil.rmtree(env.tmp_path / "run_002" / "slao")  # no accumulator to restore
        shutil.rmtree(env.tmp_path / "run_002" / "lora")  # and no weights either

        env.inner = FakeInnerTrainer()
        fake2 = make_fake_sft()
        install_fake_sft(env.monkeypatch, fake2)
        second = MultiRunTrainer(
            model="tiny-test", config=_cfg(env.tmp_path, num_runs=3), resume_from=run_id
        )

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = second.run(text_dataset(60))

        assert second._resume_slao_state_restored is False
        assert any("Failed to load resume checkpoint" in r.getMessage() for r in caplog.records)
        assert [r.run_index for r in result.runs] == [3]

    def test_resume_without_any_checkpoint_starts_again_from_run_one(self, env, caplog):
        """History says the run exists but its checkpoint manifest is empty."""
        history = RunHistoryManager(str(env.tmp_path))
        history.record_run_started(
            run_id="deadbeefcafe", model_name="m", session_kind="multi_run"
        )
        mrt = MultiRunTrainer(
            model="tiny-test", config=_cfg(env.tmp_path, num_runs=2, merge_mode=MergeMode.SIMPLE),
            resume_from="deadbeefcafe",
        )
        env.fake = make_fake_sft()
        install_fake_sft(env.monkeypatch, env.fake)

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            result = mrt.run(text_dataset(40))

        assert result.run_id == "deadbeefcafe"
        assert [r.run_index for r in result.runs] == [1, 2]  # nothing skipped
        assert any("No checkpoint found for run_id=deadbeefcafe" in r.getMessage()
                   for r in caplog.records)

    def test_unknown_resume_id_fails_before_any_work(self, env):
        mrt = MultiRunTrainer(
            model="tiny-test", config=_cfg(env.tmp_path), resume_from="no-such-run"
        )
        env.fake = make_fake_sft()
        install_fake_sft(env.monkeypatch, env.fake)

        with pytest.raises(InvalidSettingError):
            mrt.run(text_dataset(40))

        assert env.fake.created == [] and env.inner.load_model_calls == 0


# =============================================================================
# Resume-time SLAO restore failure (unit level, real merger)
# =============================================================================


class TestRestoreSessionStateFailure:
    def test_corrupt_slao_directory_is_survivable(self, env, caplog):
        """A ``slao/`` dir that cannot be loaded degrades to 're-initialise the merger'."""
        first = _crashed_session(env)
        (env.tmp_path / "run_002" / "slao" / "merged_lora.pt").write_bytes(b"corrupt")

        second = MultiRunTrainer(
            model="tiny-test", config=_cfg(env.tmp_path, num_runs=3),
            resume_from=first._run_id,
        )
        from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy
        from backpropagate.slao import SLAOMerger

        second._checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(env.tmp_path), policy=CheckpointPolicy()
        )
        second._slao_merger = SLAOMerger()

        with caplog.at_level(logging.WARNING, logger=MULTI_RUN_LOGGER):
            ok = second._restore_session_state(env.tmp_path, first._run_id)

        assert ok is True  # the checkpoint itself is still resumable
        assert second._resume_slao_state_restored is False
        assert second._slao_merger.get_merged_lora() is None
        assert any("Failed to restore SLAO state" in r.getMessage() for r in caplog.records)
