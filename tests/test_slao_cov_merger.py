"""Coverage tests for ``SLAOMerger``: merge branches, divergence detection and
checkpoint save/load failure modes.

Real tiny CPU tensors and real files under ``tmp_path`` throughout. The only
patched boundary is ``Path.mkdir`` in the single permission-denied test (a
filesystem permission cannot be provoked portably otherwise); the ``meta``
device stands in for a second compute device in the device-normalisation and
finite-probe tests.

Expected numbers are hand-computed from the SLAO rules: A hard-replaced, B
EMA-merged with lambda(i) = 1/sqrt(i) clamped to ``[min_scale, 1]``.
"""

from __future__ import annotations

import json
import logging
import math

import pytest

torch = pytest.importorskip("torch")

from backpropagate.exceptions import BackpropagateError, SLAOCheckpointError
from backpropagate.slao import (
    MergeStrategyConfig,
    SLAOConfig,
    SLAOMerger,
)

A_KEY = "base.layers.0.q.lora_A.default.weight"
B_KEY = "base.layers.0.q.lora_B.default.weight"
SLAO_LOGGER = "backpropagate.slao"


def _t(*vals: float) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32)


def _seeded(state: dict, **cfg) -> SLAOMerger:
    merger = SLAOMerger(**cfg)
    merger.initialize(state)
    return merger


# =============================================================================
# initialize / get_init_weights
# =============================================================================


class TestInitWeights:
    def test_non_tensor_entries_pass_through_and_tensors_are_cloned(self):
        """Orthogonal A (QR of A^T with sign fix) for diag rows [2,3] is the unit
        selector [[1,0,0],[0,1,0]]; B and non-tensors are carried over."""
        a = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        b = _t(5.0, 6.0)
        merger = _seeded({A_KEY: a, B_KEY: b, "tag": "meta-string"})

        init = merger.get_init_weights()

        assert init is not None
        assert init["tag"] == "meta-string"
        assert torch.allclose(init[A_KEY], torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-6)
        assert torch.equal(init[B_KEY], b)
        assert init[B_KEY] is not b  # a copy, not an alias of the caller's tensor

    def test_get_init_weights_before_initialize_is_none(self):
        assert SLAOMerger().get_init_weights() is None


# =============================================================================
# merge: non-tensor accumulators, new-key drift, history flag
# =============================================================================


class TestMergeNonTensorAccumulator:
    def test_non_tensor_a_and_b_entries_yield_no_norms_and_no_counts(self):
        """If the accumulated A/B entries are not tensors, no representative norm
        can be sampled (both before and after are None) and nothing is counted;
        the one real tensor still merges: 0 + 0.5 * (4 - 0) = 2 at run 4."""
        merger = _seeded({A_KEY: "stub", B_KEY: "stub", "misc.w": _t(0.0)})

        result = merger.merge(
            {A_KEY: "stub2", B_KEY: "stub2", "misc.w": _t(4.0)}, run_index=4
        )

        assert result.a_norm_before is None and result.a_norm_after is None
        assert result.b_norm_before is None and result.b_norm_after is None
        assert result.a_matrices_merged == 0 and result.b_matrices_merged == 0
        assert result.total_params_merged == 1
        assert torch.allclose(merger.get_merged_lora()["misc.w"], _t(2.0))


class TestNewKeyDrift:
    def test_default_path_warns_and_counts_new_keys_from_run_two(self, caplog):
        """A key unseen by the accumulator on run >= 2 is cloned in, counted in
        ``new_keys_added`` and logged as a config-drift warning."""
        merger = _seeded({B_KEY: _t(0.0, 0.0)})
        fresh = _t(3.0, 4.0)
        new_key = "base.layers.1.q.lora_B.default.weight"

        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            result = merger.merge({B_KEY: _t(1.0, 1.0), new_key: fresh}, run_index=4)

        assert result.new_keys_added == 1
        assert any("new LoRA key appeared at run=4" in r.getMessage() for r in caplog.records)
        merged = merger.get_merged_lora()
        assert torch.equal(merged[new_key], fresh)
        assert merged[new_key] is not fresh
        # B = 0 + 0.5 * (1 - 0)
        assert torch.allclose(merged[B_KEY], _t(0.5, 0.5))

    def test_linear_strategy_counts_and_warns_on_new_keys(self, caplog):
        """The non-default strategy path derives the same counters: one A, one B,
        one generic tensor merged, one brand-new key counted and cloned."""
        merger = _seeded(
            {A_KEY: _t(0.0), B_KEY: _t(0.0), "misc.w": _t(0.0)},
            strategy_config=MergeStrategyConfig(strategy="linear", linear_weight=0.5),
        )
        new_key = "base.layers.2.q.lora_B.default.weight"

        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            result = merger.merge(
                {
                    A_KEY: _t(2.0),
                    B_KEY: _t(2.0),
                    "misc.w": _t(2.0),
                    new_key: _t(7.0, 8.0),
                    "note": "ignored",
                },
                run_index=3,
            )

        assert result.strategy == "linear"
        assert result.a_matrices_merged == 1
        assert result.b_matrices_merged == 1
        assert result.new_keys_added == 1
        # 1 + 1 + 1 (common) + 2 (new key) = 5 tensor elements
        assert result.total_params_merged == 5
        assert any("(linear): new LoRA key appeared at run=3" in r.getMessage() for r in caplog.records)
        merged = merger.get_merged_lora()
        # linear blends A too (no hard replace): 0 + 0.5 * 2 = 1
        assert torch.allclose(merged[A_KEY], _t(1.0))
        assert torch.allclose(merged["misc.w"], _t(1.0))
        assert torch.equal(merged[new_key], _t(7.0, 8.0))

    def test_strategy_path_keeps_accumulator_only_non_tensor_entries(self):
        """The strategy path's finite scan skips non-tensor entries that were
        cloned through from the accumulator."""
        merger = _seeded(
            {"misc.w": _t(0.0), "tag": "kept"},
            strategy_config=MergeStrategyConfig(strategy="linear", linear_weight=0.5),
        )
        merger.merge({"misc.w": _t(2.0)}, run_index=2)
        merged = merger.get_merged_lora()
        assert merged["tag"] == "kept"
        assert torch.allclose(merged["misc.w"], _t(1.0))


class TestMergeHistoryFlag:
    def test_history_is_not_recorded_when_disabled(self):
        merger = _seeded(
            {B_KEY: _t(0.0)}, config=SLAOConfig(save_merge_history=False)
        )
        result = merger.merge({B_KEY: _t(1.0)}, run_index=4)
        assert result.run_index == 4
        assert merger.merge_history == []


# =============================================================================
# merge: divergence detection
# =============================================================================


class TestDivergenceDetection:
    @pytest.mark.parametrize(
        "key, kind",
        [
            (A_KEY, "lora_A"),
            (B_KEY, "lora_B"),
            ("misc.w", "other"),
        ],
    )
    def test_strategy_path_reports_first_non_finite_key_and_kind(self, key, kind):
        """Linear blend of NaN yields NaN; the error names the key and its kind."""
        merger = _seeded(
            {key: _t(0.0)},
            strategy_config=MergeStrategyConfig(strategy="linear", linear_weight=0.5),
        )
        with pytest.raises(BackpropagateError) as exc_info:
            merger.merge({key: _t(float("nan"))}, run_index=2, run_id="rid-1")

        err = exc_info.value
        assert err.code == "SLAO_MERGE_DIVERGED"
        assert err.details["layer"] == key
        assert err.details["kind"] == kind
        assert err.details["run_id"] == "rid-1"
        assert err.details["run_index"] == 2
        assert err.details["scan_mode"] == "full"
        assert err.retryable is False

    def test_default_path_reports_only_the_first_of_several_bad_keys(self):
        """Two NaN tensors: the first iterated key is the one reported."""
        merger = _seeded({A_KEY: _t(0.0), B_KEY: _t(0.0)})
        with pytest.raises(BackpropagateError) as exc_info:
            merger.merge(
                {A_KEY: _t(float("nan")), B_KEY: _t(float("nan"))}, run_index=2
            )
        assert exc_info.value.details["layer"] == A_KEY
        assert exc_info.value.details["kind"] == "lora_A"

    def test_a_norm_overflow_is_promoted_to_divergence(self):
        """[3e38, 3e38] is finite in float32 but its L2 norm overflows to inf. The
        per-key finite probe passes, so the representative-norm backstop fires."""
        big = _t(3e38, 3e38)
        merger = _seeded({A_KEY: _t(1.0, 1.0)})
        with pytest.raises(BackpropagateError) as exc_info:
            merger.merge({A_KEY: big}, run_index=2)

        details = exc_info.value.details
        assert exc_info.value.code == "SLAO_MERGE_DIVERGED"
        assert details["layer"] == A_KEY
        assert details["kind"] == "lora_A"
        assert math.isinf(details["a_norm_after"])

    def test_b_norm_overflow_is_promoted_to_divergence(self):
        """acc == new == [3e38, 3e38] keeps B finite (delta 0) but its norm is inf;
        A stays small so the lora_B branch of the backstop is the one taken."""
        big = _t(3e38, 3e38)
        merger = _seeded({A_KEY: _t(1.0), B_KEY: big.clone()})
        with pytest.raises(BackpropagateError) as exc_info:
            merger.merge({A_KEY: _t(1.0), B_KEY: big.clone()}, run_index=2)

        details = exc_info.value.details
        assert details["layer"] == B_KEY
        assert details["kind"] == "lora_B"
        assert math.isinf(details["b_norm_after"])
        assert math.isfinite(details["a_norm_after"])


class TestFiniteProbeOnMetaDevice:
    """``Tensor.item()`` raises RuntimeError on the meta device, which is the
    probe-failure path: the merge must succeed and only log at DEBUG."""

    def test_default_path_survives_probe_runtime_error(self, caplog):
        merger = _seeded({"misc.w": _t(0.0, 0.0)})
        with caplog.at_level(logging.DEBUG, logger=SLAO_LOGGER):
            result = merger.merge(
                {"misc.w": torch.zeros(2, device="meta")}, run_index=2
            )

        assert result.total_params_merged == 2
        assert merger.get_merged_lora()["misc.w"].device.type == "meta"
        assert any("finite-probe failed for misc.w" in r.getMessage() for r in caplog.records)

    def test_strategy_path_survives_probe_runtime_error(self, caplog):
        merger = _seeded(
            {"misc.w": _t(0.0, 0.0)},
            strategy_config=MergeStrategyConfig(strategy="linear", linear_weight=0.5),
        )
        with caplog.at_level(logging.DEBUG, logger=SLAO_LOGGER):
            result = merger.merge(
                {"misc.w": torch.zeros(2, device="meta")}, run_index=2
            )

        assert result.total_params_merged == 2
        assert any("finite-probe failed for misc.w" in r.getMessage() for r in caplog.records)


# =============================================================================
# save: failure modes
# =============================================================================


def _merger_with_state() -> SLAOMerger:
    merger = _seeded({A_KEY: _t(1.0, 2.0), B_KEY: _t(3.0, 4.0)})
    merger.merge({A_KEY: _t(5.0, 6.0), B_KEY: _t(7.0, 8.0)}, run_index=2)
    return merger


class TestSaveEdges:
    def test_save_without_state_writes_history_only(self, tmp_path):
        target = tmp_path / "ck"
        SLAOMerger().save(str(target))

        assert not (target / "merged_lora.pt").exists()
        data = json.loads((target / "merge_history.json").read_text())
        assert data["version"] == SLAOMerger.CURRENT_SLAO_VERSION
        assert data["history"] == []
        assert data["run_index"] == 0

    def test_resave_replaces_previous_directory(self, tmp_path):
        target = tmp_path / "ck"
        merger = _merger_with_state()
        merger.save(str(target))
        (target / "stale.txt").write_text("from the first save")

        merger.merge({A_KEY: _t(9.0, 9.0), B_KEY: _t(9.0, 9.0)}, run_index=3)
        merger.save(str(target), run_id="second")

        assert not (target / "stale.txt").exists()
        data = json.loads((target / "merge_history.json").read_text())
        assert data["run_index"] == 3
        assert data["run_id"] == "second"
        assert [h["run_index"] for h in data["history"]] == [2, 3]
        assert not (tmp_path / "ck.partial").exists()

    def test_leftover_partial_directory_from_a_crash_is_wiped(self, tmp_path):
        target = tmp_path / "ck"
        partial = tmp_path / "ck.partial"
        partial.mkdir()
        (partial / "junk.bin").write_text("half written")

        _merger_with_state().save(str(target))

        assert not partial.exists()
        assert not (target / "junk.bin").exists()
        assert (target / "merged_lora.pt").exists()

    def test_parent_that_is_a_file_raises_checkpoint_error(self, tmp_path):
        blocker = tmp_path / "iamafile"
        blocker.write_text("x")
        with pytest.raises(SLAOCheckpointError) as exc_info:
            _merger_with_state().save(str(blocker / "ck"))

        assert exc_info.value.code == "STATE_SLAO_CHECKPOINT_INVALID"
        assert exc_info.value.operation == "save"
        assert "Failed to create parent directory" in str(exc_info.value)

    def test_permission_denied_on_parent_is_reported(self, tmp_path, monkeypatch):
        """Mocked boundary: ``Path.mkdir`` raising PermissionError (portable
        stand-in for a read-only filesystem)."""
        from pathlib import Path

        def deny(self, *args, **kwargs):
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(Path, "mkdir", deny)
        with pytest.raises(SLAOCheckpointError) as exc_info:
            _merger_with_state().save(str(tmp_path / "sub" / "ck"))

        assert "Permission denied creating parent directory" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, PermissionError)

    def test_partial_slot_blocked_by_a_file_is_reported(self, tmp_path):
        """A *file* named ``ck.partial`` survives the best-effort cleanup, so the
        partial directory cannot be created."""
        (tmp_path / "ck.partial").write_text("not a directory")
        with pytest.raises(SLAOCheckpointError) as exc_info:
            _merger_with_state().save(str(tmp_path / "ck"))

        assert "Failed to create partial directory" in str(exc_info.value)
        assert not (tmp_path / "ck").exists()

    def test_unpicklable_weights_leave_no_partial_or_final_dir(self, tmp_path):
        merger = _merger_with_state()
        merger._merged_state["bad"] = lambda: None  # real torch.save cannot pickle this

        with pytest.raises(SLAOCheckpointError) as exc_info:
            merger.save(str(tmp_path / "ck"))

        assert "Failed to save weights" in str(exc_info.value)
        assert not (tmp_path / "ck").exists()
        assert not (tmp_path / "ck.partial").exists()

    def test_unserialisable_history_leaves_no_partial_or_final_dir(self, tmp_path):
        """``run_id`` is persisted verbatim; an object json cannot encode fails the
        history write after the weights were already written."""
        with pytest.raises(SLAOCheckpointError) as exc_info:
            _merger_with_state().save(str(tmp_path / "ck"), run_id=object())  # type: ignore[arg-type]

        assert "Failed to save history" in str(exc_info.value)
        assert not (tmp_path / "ck").exists()
        assert not (tmp_path / "ck.partial").exists()

    def test_promotion_failure_is_wrapped_and_cleaned_up(self, tmp_path):
        """If the destination exists as a *file*, removing it for the promote step
        fails; the error is wrapped and the partial directory removed."""
        target = tmp_path / "ck"
        target.write_text("squatting file")

        with pytest.raises(SLAOCheckpointError) as exc_info:
            _merger_with_state().save(str(target))

        assert "Atomic promotion failed" in str(exc_info.value)
        assert not (tmp_path / "ck.partial").exists()
        assert target.read_text() == "squatting file"


# =============================================================================
# load: failure modes and version / default fallbacks
# =============================================================================


def _saved(tmp_path):
    merger = _merger_with_state()
    target = tmp_path / "ck"
    merger.save(str(target))
    return target


class TestLoadFailures:
    def test_missing_directory(self, tmp_path):
        with pytest.raises(SLAOCheckpointError) as exc_info:
            SLAOMerger().load(str(tmp_path / "nope"))
        assert "Checkpoint directory not found" in str(exc_info.value)

    def test_missing_weights_file(self, tmp_path):
        (tmp_path / "ck").mkdir()
        with pytest.raises(SLAOCheckpointError) as exc_info:
            SLAOMerger().load(str(tmp_path / "ck"))
        assert "No merged_lora.pt found" in str(exc_info.value)

    def test_garbage_weights_file(self, tmp_path):
        ck = tmp_path / "ck"
        ck.mkdir()
        (ck / "merged_lora.pt").write_bytes(b"this is not a torch checkpoint")
        with pytest.raises(SLAOCheckpointError) as exc_info:
            SLAOMerger().load(str(ck))
        assert "Failed to load weights" in str(exc_info.value)


class TestLoadHistoryFallbacks:
    def _rewrite_history(self, ck, mutate):
        path = ck / "merge_history.json"
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    def test_schema_version_mismatch_warns_but_still_restores(self, tmp_path, caplog):
        ck = _saved(tmp_path)
        self._rewrite_history(ck, lambda d: d.update(version="0.9"))

        merger = SLAOMerger()
        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            merger.load(str(ck))

        assert any("version='0.9'" in r.getMessage() for r in caplog.records)
        assert merger.run_index == 2
        assert torch.equal(merger.get_merged_lora()[A_KEY], _t(5.0, 6.0))

    def test_pre_v13_checkpoint_warns_per_missing_field_and_keeps_live_values(
        self, tmp_path, caplog
    ):
        ck = _saved(tmp_path)

        def to_old(d):
            d["version"] = "1.0"
            d["run_index"] = 3
            d["config"] = {
                "scaling_type": "linear",
                "min_scale": 0.2,
                "use_orthogonal_init": False,
            }
            d.pop("strategy_config")

        self._rewrite_history(ck, to_old)
        merger = SLAOMerger(config=SLAOConfig(use_layer_scaling=True, layer_scale_early=0.9))
        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            merger.load(str(ck))

        msgs = [r.getMessage() for r in caplog.records]
        for field in (
            "use_time_aware_scaling",
            "normalize_after_merge",
            "use_adaptive_scaling",
            "use_layer_scaling",
            "layer_scale_early",
            "layer_scale_middle",
            "layer_scale_late",
        ):
            assert any(f"{field!r}" in m and "pre-v1.3" in m for m in msgs), field
        assert any("pre-1.1 SLAO checkpoint" in m for m in msgs)

        # on-disk fields restored; missing ones keep the live runtime value
        assert merger.config.scaling_type == "linear"
        assert merger.config.min_scale == 0.2
        assert merger.config.use_orthogonal_init is False
        assert merger.config.use_layer_scaling is True
        assert merger.config.layer_scale_early == 0.9
        assert merger.run_index == 3
        assert merger.strategy_config.strategy == "qiao_mahdavi"

    def test_valid_adaptive_range_is_restored_as_float_tuple(self, tmp_path):
        ck = _saved(tmp_path)
        self._rewrite_history(ck, lambda d: d["config"].update(adaptive_scale_range=[1, 2]))
        merger = SLAOMerger()
        merger.load(str(ck))
        assert merger.config.adaptive_scale_range == (1.0, 2.0)

    @pytest.mark.parametrize("bad", [[0.5, 1.0, 1.5], "oops", [1.0]])
    def test_malformed_adaptive_range_keeps_the_live_value(self, tmp_path, bad):
        ck = _saved(tmp_path)
        self._rewrite_history(ck, lambda d: d["config"].update(adaptive_scale_range=bad))
        merger = SLAOMerger(config=SLAOConfig(adaptive_scale_range=(0.7, 1.3)))
        merger.load(str(ck))
        assert merger.config.adaptive_scale_range == (0.7, 1.3)

    def test_corrupt_history_json_keeps_weights_and_warns(self, tmp_path, caplog):
        ck = _saved(tmp_path)
        (ck / "merge_history.json").write_text("{this is not json")

        merger = SLAOMerger()
        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            merger.load(str(ck))

        assert any("merge_history.json is corrupted" in r.getMessage() for r in caplog.records)
        # run 2 merge on disk: B = [3,4] + (1/sqrt(2)) * ([7,8] - [3,4])
        expected_b = _t(3.0 + 4.0 / math.sqrt(2), 4.0 + 4.0 / math.sqrt(2))
        assert torch.allclose(merger.get_merged_lora()[B_KEY], expected_b, atol=1e-5)
        assert merger.run_index == 0

    def test_history_with_unexpected_shape_keeps_weights_and_warns(self, tmp_path, caplog):
        """A JSON list (not an object) raises inside the restore block; the
        generic handler logs it and the loaded weights stay usable."""
        ck = _saved(tmp_path)
        (ck / "merge_history.json").write_text("[]")

        merger = SLAOMerger()
        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            merger.load(str(ck))

        assert any("Failed to load merge_history.json" in r.getMessage() for r in caplog.records)
        assert set(merger.get_merged_lora()) == {A_KEY, B_KEY}


class TestRunOneKeyDriftIsSilent:
    """A key unseen by the accumulator is only drift from run 2 onward."""

    @pytest.mark.parametrize("strategy", ["qiao_mahdavi", "linear"])
    def test_run_index_one_does_not_count_new_keys(self, strategy, caplog):
        merger = _seeded(
            {B_KEY: _t(0.0)},
            strategy_config=MergeStrategyConfig(strategy=strategy, linear_weight=0.5),
        )
        new_key = "base.layers.1.q.lora_B.default.weight"
        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            result = merger.merge({B_KEY: _t(2.0), new_key: _t(5.0)}, run_index=1)

        assert result.new_keys_added == 0
        assert not any("new LoRA key" in r.getMessage() for r in caplog.records)
        assert torch.equal(merger.get_merged_lora()[new_key], _t(5.0))


class TestLoadWithoutOptionalFiles:
    def test_checkpoint_without_history_file_loads_weights_only(self, tmp_path):
        ck = _saved(tmp_path)
        (ck / "merge_history.json").unlink()

        merger = SLAOMerger()
        merger.load(str(ck))

        assert merger.run_index == 0
        assert torch.equal(merger.get_merged_lora()[A_KEY], _t(5.0, 6.0))

    def test_current_version_without_strategy_block_does_not_warn(self, tmp_path, caplog):
        ck = _saved(tmp_path)
        path = ck / "merge_history.json"
        data = json.loads(path.read_text())
        data.pop("strategy_config")
        path.write_text(json.dumps(data))

        merger = SLAOMerger(strategy_config=MergeStrategyConfig(strategy="linear"))
        with caplog.at_level(logging.WARNING, logger=SLAO_LOGGER):
            merger.load(str(ck))

        assert not any("pre-1.1" in r.getMessage() for r in caplog.records)
        assert merger.strategy_config.strategy == "linear"  # live value kept
