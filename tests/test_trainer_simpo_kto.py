"""Tests for v1.6 C2 — SimPO + KTO (trainer-integration / Wave 2).

SimPO and KTO are the two new preference objectives in v1.6. Wave 1 landed the
config fields (``simpo_beta``/``simpo_gamma``/``kto_*``), the ``KTO`` dataset
format + ``to_kto_dataset``, and the ``get_recommended_lr`` ladder. This file
pins the *trainer wiring* (the Wave-2 half), exactly mirroring
``tests/test_orpo.py`` for the ORPO objective:

- ``Trainer(method='simpo'|'kto')`` resolution + validation + the +full guards.
- The SimPO LR default (1e-6) + high-LR clamp; the KTO LR default (1e-6) +
  >5e-6 clamp; explicit-LR honoring.
- ``_build_cpo_config`` — SimPO selection (``loss_type='simpo'`` +
  ``cpo_alpha=0.0`` FORCED), ``beta``/``simpo_gamma``/``max_length`` mapping, no
  SFT-only ``packing``, and the CPU-runner regression guard (CPU → adamw_torch
  + fp32) plus the consumer-GPU paged/bf16 path.
- ``_build_kto_config`` — ``beta``/weights/``max_length`` mapping +
  ``train_sampling_strategy='sequential'`` (FORCED for the KL estimate).
- ``train()`` objective dispatch — SimPO builds a ``CPOTrainer`` and KTO a
  ``KTOTrainer`` (NEITHER an ``SFTTrainer``), both skip
  ``_apply_train_on_responses_only`` + the Windows pre-tokenize, tag run-history
  with ``method`` + the objective knobs, and the OOM-retry path REBUILDS the
  same preference trainer via the shared ``_build_trainer`` helper (the
  BACKEND-A-003 anti-drift contract, extended to SimPO/KTO).
- KTO passes NO explicit ``ref_model`` (the frozen LoRA base is the reference —
  a second model would break the 16GB envelope, design-lock C2).
- KTO auto-weighting — counts labels, rebalances toward [1:1, 4:3].
- ``_load_dataset(method='simpo')`` uses the PAIRED path (``to_preference_
  dataset``, identical to ORPO); ``method='kto'`` uses the UNPAIRED path
  (``to_kto_dataset``); each raises ``DatasetFormatError`` on the wrong shape.

All tests are pure-Python / CPU / trl-MOCKED (the real-train proofs are the
non-mocked smokes ``test_simpo_smoke.py`` / ``test_kto_smoke.py``). They follow
the ``patch.dict("sys.modules", {"trl": MagicMock(...)})`` +
``patch("torch.cuda.is_available", ...)`` patterns established in
``test_orpo.py`` / ``test_wave6b_features.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Shared stubs / helpers
# =============================================================================


class _StubConfig:
    """Stub trl config that records the kwargs it was constructed with.

    Mirrors ``_StubORPOConfig`` in test_orpo.py — any object capturing
    **kwargs works; the stub means the config-shape assertions do not depend on
    a real trl install.
    """

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


def _build_cpo_config_capturing(**call_kwargs) -> dict:
    """Call ``_build_cpo_config`` with trl mocked, returning captured kwargs."""
    from backpropagate import trainer as trainer_mod

    captured: dict = {}

    class _Capture(_StubConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    with patch.dict("sys.modules", {"trl": MagicMock(CPOConfig=_Capture)}):
        trainer_mod._build_cpo_config(**call_kwargs)
    return captured


def _build_kto_config_capturing(**call_kwargs) -> dict:
    """Call ``_build_kto_config`` with trl mocked, returning captured kwargs."""
    from backpropagate import trainer as trainer_mod

    captured: dict = {}

    class _Capture(_StubConfig):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    with patch.dict("sys.modules", {"trl": MagicMock(KTOConfig=_Capture)}):
        trainer_mod._build_kto_config(**call_kwargs)
    return captured


_CPO_CFG_BASE: dict = {
    "output_dir": "./out",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 10,
    "learning_rate": 1e-6,
    "warmup_steps": 2,
    "max_seq_length": 1024,
    "simpo_beta": 2.0,
    "simpo_gamma": 1.0,
    "seed": 42,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
}

_KTO_CFG_BASE: dict = {
    "output_dir": "./out",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 10,
    "learning_rate": 1e-6,
    "warmup_steps": 2,
    "max_seq_length": 1024,
    "kto_beta": 0.1,
    "desirable_weight": 1.0,
    "undesirable_weight": 1.0,
    "seed": 42,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
}


class _OOMScript:
    """Preference-trainer mock factory whose .train() OOMs N times then succeeds.

    Mirrors test_orpo.py::_ORPOOOMScript: the OOM-recovery loop re-instantiates
    the inner trainer on each retry, so we need a fresh mock per construction
    that still shares the failure counter, and we count constructions so a test
    can assert the retry REBUILT the SAME preference trainer (the anti-drift
    contract extended to SimPO/KTO).
    """

    def __init__(self, oom_count: int):
        self._oom_remaining = oom_count
        self.train_calls = 0
        self.construct_calls = 0

    def factory(self, *args, **kwargs):
        self.construct_calls += 1
        instance = MagicMock()

        def train_impl(*a, **k):
            self.train_calls += 1
            if self._oom_remaining > 0:
                self._oom_remaining -= 1
                raise RuntimeError("CUDA out of memory at batch_size attempt")
            result = MagicMock()
            result.training_loss = 0.42
            return result

        instance.train.side_effect = train_impl
        instance.state.log_history = [{"loss": 0.42}]
        return instance


def _mock_pref_instance(training_loss: float = 0.5):
    """A preference-trainer mock whose .train() returns a finite loss once."""
    inst = MagicMock()
    result = MagicMock()
    result.training_loss = training_loss
    inst.train.return_value = result
    inst.state.log_history = [{"loss": training_loss}]
    return inst


def _pref_trainer_ready(temp_dir, method: str, **kwargs):
    """Construct a CPU Trainer(method=...) with a fake loaded model."""
    from backpropagate.trainer import Trainer

    # Pin backend so 'auto' never routes to the MLX rail on an Apple-Silicon CI
    # runner (preference objectives are blocked on MLX).
    kwargs.setdefault("backend", "cuda")
    with patch("torch.cuda.is_available", return_value=False):
        trainer = Trainer(
            method=method,
            output_dir=str(temp_dir),
            use_unsloth=False,
            mode="lora",
            **kwargs,
        )
    trainer._model = MagicMock()
    trainer._tokenizer = MagicMock()
    trainer._is_loaded = True
    return trainer


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


# =============================================================================
# Trainer construction: method resolution + guards
# =============================================================================


class TestSimpoKtoMethodResolution:
    """v1.6 C2: Trainer(method='simpo'|'kto') resolution, validation, guards."""

    @pytest.mark.parametrize("method", ["simpo", "kto"])
    def test_method_sets_attribute(self, method):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method=method, mode="lora")
        assert trainer.method == method

    def test_simpo_default_knobs(self):
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="simpo", mode="lora")
        assert trainer.simpo_beta == pytest.approx(settings.training.simpo_beta)
        assert trainer.simpo_gamma == pytest.approx(settings.training.simpo_gamma)

    def test_simpo_explicit_knobs(self):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="simpo", mode="lora", simpo_beta=2.5, simpo_gamma=1.2
            )
        assert trainer.simpo_beta == pytest.approx(2.5)
        assert trainer.simpo_gamma == pytest.approx(1.2)

    def test_kto_default_knobs(self):
        from backpropagate.config import settings
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="kto", mode="lora")
        assert trainer.kto_beta == pytest.approx(settings.training.kto_beta)
        assert trainer.kto_desirable_weight == pytest.approx(
            settings.training.kto_desirable_weight
        )
        assert trainer.kto_undesirable_weight == pytest.approx(
            settings.training.kto_undesirable_weight
        )

    def test_kto_explicit_knobs(self):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="kto",
                mode="lora",
                kto_beta=0.2,
                kto_desirable_weight=1.5,
                kto_undesirable_weight=2.0,
            )
        assert trainer.kto_beta == pytest.approx(0.2)
        assert trainer.kto_desirable_weight == pytest.approx(1.5)
        assert trainer.kto_undesirable_weight == pytest.approx(2.0)

    def test_invalid_method_raises(self):
        """An unknown method raises InvalidSettingError (defense in depth)."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError):
                Trainer(method="dpo", mode="lora")

    @pytest.mark.parametrize("method", ["simpo", "kto"])
    def test_method_plus_full_raises(self, method):
        """simpo/kto + mode='full' is blocked at construction (lora-only)."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as exc:
                Trainer(method=method, mode="full")
        # The error names the method+mode pair.
        assert method in str(exc.value)

    @pytest.mark.parametrize("bad_gamma", [0.0, -1.0])
    def test_simpo_nonpositive_gamma_raises(self, bad_gamma):
        """A DIRECT Trainer(method='simpo', simpo_gamma<=0) raises."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError):
                Trainer(method="simpo", mode="lora", simpo_gamma=bad_gamma)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("kto_beta", 0.0),
            ("kto_beta", -0.1),
            ("kto_desirable_weight", 0.0),
            ("kto_undesirable_weight", -1.0),
        ],
    )
    def test_kto_nonpositive_knob_raises(self, field, value):
        """A DIRECT Trainer(method='kto', <knob><=0) raises."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError):
                Trainer(method="kto", mode="lora", **{field: value})

    @pytest.mark.parametrize("method", ["simpo", "kto"])
    def test_method_plus_fp8_raises(self, method):
        """simpo/kto + fp8 is blocked (FP8 validated for sft only)."""
        from backpropagate.exceptions import InvalidSettingError
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError):
                Trainer(method=method, mode="lora", fp8=True)

    def test_kto_batch_size_one_bumped_to_two(self):
        """method='kto' with batch_size=1 auto-bumps to 2.

        Regression guard for the constraint the non-mocked KTO smoke surfaced:
        KTOTrainer rejects an actual per-device batch size of 1 (the in-batch
        KL term collapses to the implied reward). The trainer bumps the floor
        to 2 with a warning rather than crashing deep inside trl.
        """
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="kto", mode="lora", batch_size=1)
        assert trainer.batch_size == 2

    def test_non_kto_batch_size_one_preserved(self):
        """SFT/SimPO keep batch_size=1 (the KTO floor is method-scoped)."""
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            sft = Trainer(method="sft", mode="lora", batch_size=1)
            simpo = Trainer(method="simpo", mode="lora", batch_size=1)
        assert sft.batch_size == 1
        assert simpo.batch_size == 1


# =============================================================================
# Learning-rate defaults + clamps
# =============================================================================


class TestSimpoKtoLearningRate:
    """v1.6 C2: SimPO/KTO LR auto-lower + high-LR clamp."""

    def test_simpo_default_lr(self):
        from backpropagate.trainer import Trainer, _SIMPO_DEFAULT_LR

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="simpo", mode="lora")
        assert trainer.learning_rate == pytest.approx(_SIMPO_DEFAULT_LR)

    def test_simpo_explicit_low_lr_honored(self):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="simpo", mode="lora", learning_rate=5e-6)
        assert trainer.learning_rate == pytest.approx(5e-6)

    @pytest.mark.parametrize("high_lr", [1e-5, 2e-4])
    def test_simpo_high_lr_clamped(self, high_lr):
        """SimPO LR >= 1e-5 clamps down to the stable anchor."""
        from backpropagate.trainer import Trainer, _SIMPO_DEFAULT_LR

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="simpo", mode="lora", learning_rate=high_lr)
        assert trainer.learning_rate == pytest.approx(_SIMPO_DEFAULT_LR)

    def test_kto_default_lr(self):
        from backpropagate.trainer import Trainer, _KTO_DEFAULT_LR

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="kto", mode="lora")
        assert trainer.learning_rate == pytest.approx(_KTO_DEFAULT_LR)

    def test_kto_explicit_low_lr_honored(self):
        from backpropagate.trainer import Trainer

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="kto", mode="lora", learning_rate=3e-6)
        assert trainer.learning_rate == pytest.approx(3e-6)

    @pytest.mark.parametrize("high_lr", [6e-6, 1e-5, 2e-4])
    def test_kto_high_lr_clamped(self, high_lr):
        """KTO LR > 5e-6 clamps down to the published anchor."""
        from backpropagate.trainer import Trainer, _KTO_DEFAULT_LR

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(method="kto", mode="lora", learning_rate=high_lr)
        assert trainer.learning_rate == pytest.approx(_KTO_DEFAULT_LR)


# =============================================================================
# Config builders: _build_cpo_config (SimPO) + _build_kto_config
# =============================================================================


class TestBuildCpoConfig:
    """v1.6 C2: _build_cpo_config produces a pure-SimPO CPOConfig."""

    def test_simpo_selection_forced(self):
        """loss_type='simpo' + cpo_alpha=0.0 are always set (pure SimPO)."""
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(**_CPO_CFG_BASE)
        assert captured["loss_type"] == "simpo"
        assert captured["cpo_alpha"] == 0.0, (
            "cpo_alpha MUST be forced to 0.0 — a non-zero value is CPO-SimPO, a "
            "different method."
        )

    def test_simpo_beta_and_gamma_mapped(self):
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(
                **{**_CPO_CFG_BASE, "simpo_beta": 2.5, "simpo_gamma": 1.4}
            )
        assert captured["beta"] == pytest.approx(2.5)
        assert captured["simpo_gamma"] == pytest.approx(1.4)

    def test_max_seq_length_maps_to_max_length(self):
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(
                **{**_CPO_CFG_BASE, "max_seq_length": 512}
            )
        assert captured["max_length"] == 512

    def test_no_packing_field(self):
        """CPOConfig has no 'packing' (it rejects unknown kwargs)."""
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(**_CPO_CFG_BASE)
        assert "packing" not in captured

    def test_cpu_runner_uses_adamw_torch_and_fp32(self):
        """CPU regression guard: optim downgrades + bf16/fp16 both False."""
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(**_CPO_CFG_BASE)
        assert captured["optim"] == "adamw_torch"
        assert captured["bf16"] is False
        assert captured["fp16"] is False

    def test_optional_truncation_knobs_added_only_when_set(self):
        with patch("torch.cuda.is_available", return_value=False):
            without = _build_cpo_config_capturing(**_CPO_CFG_BASE)
            with_knobs = _build_cpo_config_capturing(
                **{
                    **_CPO_CFG_BASE,
                    "max_prompt_length": 64,
                    "max_completion_length": 128,
                }
            )
        assert "max_prompt_length" not in without
        assert with_knobs["max_prompt_length"] == 64
        assert with_knobs["max_completion_length"] == 128

    def test_small_max_seq_length_derives_safe_max_prompt_length(self):
        """max_seq_length <= 512 derives max_prompt_length so CPOTrainer's
        ``max_prompt_length < max_length`` invariant holds.

        Regression guard for the footgun the non-mocked SimPO smoke surfaced:
        CPOConfig's default max_prompt_length is 512, so a 128 window crashed
        CPOTrainer with "max_prompt_length (512) should be strictly less than
        max_length (128)".
        """
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(
                **{**_CPO_CFG_BASE, "max_seq_length": 128}
            )
        assert captured["max_length"] == 128
        assert captured["max_prompt_length"] == 64  # 128 // 2
        assert captured["max_prompt_length"] < captured["max_length"]

    def test_explicit_max_prompt_length_wins_over_derivation(self):
        """An explicit max_prompt_length is honored even on a small window."""
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_cpo_config_capturing(
                **{**_CPO_CFG_BASE, "max_seq_length": 128, "max_prompt_length": 32}
            )
        assert captured["max_prompt_length"] == 32

    def test_import_failure_raises_structured_error(self):
        """A trl with no CPOConfig (top-level OR experimental) → TrainingError."""
        from backpropagate import trainer as trainer_mod
        from backpropagate.exceptions import TrainingError

        # A trl mock that has neither trl.CPOConfig nor trl.experimental.cpo.
        broken = MagicMock(spec=[])  # spec=[] → getattr raises AttributeError
        with patch.dict("sys.modules", {"trl": broken}):
            # Also ensure the experimental submodule import fails.
            with patch.dict("sys.modules", {"trl.experimental.cpo": None}):
                with patch("torch.cuda.is_available", return_value=False):
                    with pytest.raises(TrainingError) as exc:
                        trainer_mod._build_cpo_config(**_CPO_CFG_BASE)
        assert exc.value.code == "RUNTIME_TRAINING_FAILED"


class TestBuildKtoConfig:
    """v1.6 C2: _build_kto_config produces a KTOConfig with sequential KL."""

    def test_beta_and_weights_mapped(self):
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_kto_config_capturing(
                **{
                    **_KTO_CFG_BASE,
                    "kto_beta": 0.2,
                    "desirable_weight": 1.5,
                    "undesirable_weight": 2.0,
                }
            )
        assert captured["beta"] == pytest.approx(0.2)
        assert captured["desirable_weight"] == pytest.approx(1.5)
        assert captured["undesirable_weight"] == pytest.approx(2.0)

    def test_sequential_sampling_forced(self):
        """train_sampling_strategy='sequential' is required for the KL estimate."""
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_kto_config_capturing(**_KTO_CFG_BASE)
        assert captured["train_sampling_strategy"] == "sequential"

    def test_max_seq_length_maps_to_max_length(self):
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_kto_config_capturing(
                **{**_KTO_CFG_BASE, "max_seq_length": 256}
            )
        assert captured["max_length"] == 256

    def test_cpu_runner_uses_adamw_torch_and_fp32(self):
        with patch("torch.cuda.is_available", return_value=False):
            captured = _build_kto_config_capturing(**_KTO_CFG_BASE)
        assert captured["optim"] == "adamw_torch"
        assert captured["bf16"] is False
        assert captured["fp16"] is False

    def test_import_failure_raises_structured_error(self):
        from backpropagate import trainer as trainer_mod
        from backpropagate.exceptions import TrainingError

        broken = MagicMock(spec=[])
        with patch.dict("sys.modules", {"trl": broken}):
            with patch.dict("sys.modules", {"trl.experimental.kto": None}):
                with patch("torch.cuda.is_available", return_value=False):
                    with pytest.raises(TrainingError) as exc:
                        trainer_mod._build_kto_config(**_KTO_CFG_BASE)
        assert exc.value.code == "RUNTIME_TRAINING_FAILED"


# =============================================================================
# train() objective dispatch — _build_trainer routing
# =============================================================================


class TestSimpoDispatch:
    """v1.6 C2: train(method='simpo') builds a CPOTrainer, not SFTTrainer."""

    def test_builds_cpo_trainer_not_sft(self, temp_dir):
        trainer = _pref_trainer_ready(temp_dir, "simpo")
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        cpo_inst = _mock_pref_instance()

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.CPOTrainer", return_value=cpo_inst
        ) as m_cpo, patch("trl.SFTTrainer") as m_sft, patch("trl.CPOConfig"):
            run = trainer.train("dummy", steps=5)

        assert m_cpo.called, "CPOTrainer should be constructed for method='simpo'."
        assert not m_sft.called, "SFTTrainer must NOT be constructed for SimPO."
        assert run is not None

    def test_skips_train_on_responses_only(self, temp_dir):
        from backpropagate import trainer as trainer_mod

        trainer = _pref_trainer_ready(temp_dir, "simpo")
        trainer.use_unsloth = True  # prove the gate is the method, not unsloth
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.CPOTrainer", return_value=_mock_pref_instance()
        ), patch("trl.SFTTrainer"), patch("trl.CPOConfig"), patch.object(
            trainer_mod, "_apply_train_on_responses_only"
        ) as m_apply:
            trainer.train("dummy", steps=5)

        assert not m_apply.called

    def test_records_method_and_knobs_in_run_history(self, temp_dir):
        from backpropagate.checkpoints import RunHistoryManager

        trainer = _pref_trainer_ready(
            temp_dir, "simpo", simpo_beta=2.5, simpo_gamma=1.2
        )
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.CPOTrainer", return_value=_mock_pref_instance()
        ), patch("trl.SFTTrainer"), patch("trl.CPOConfig"):
            run = trainer.train("dummy", steps=5)

        hp = RunHistoryManager(str(temp_dir)).get_run(run.run_id).get(
            "hyperparameters", {}
        )
        assert hp.get("method") == "simpo"
        assert hp.get("simpo_beta") == pytest.approx(2.5)
        assert hp.get("simpo_gamma") == pytest.approx(1.2)

    def test_oom_retry_rebuilds_cpo_trainer(self, temp_dir):
        """One OOM → the retry REBUILDS a CPOTrainer (not an SFTTrainer)."""
        trainer = _pref_trainer_ready(temp_dir, "simpo")
        trainer.oom_recovery = True
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        script = _OOMScript(oom_count=1)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch(
            "trl.CPOTrainer", side_effect=script.factory
        ), patch("trl.SFTTrainer") as m_sft, patch("trl.CPOConfig"):
            trainer.train("dummy", steps=5)

        assert script.construct_calls >= 2, (
            "Expected the CPOTrainer to be rebuilt on the OOM retry; "
            f"construct_calls={script.construct_calls}."
        )
        assert not m_sft.called, "OOM retry must not silently fall back to SFT."


class TestKtoDispatch:
    """v1.6 C2: train(method='kto') builds a KTOTrainer with no ref_model."""

    def test_builds_kto_trainer_not_sft(self, temp_dir):
        trainer = _pref_trainer_ready(temp_dir, "kto")
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch.object(
            trainer, "_auto_balance_kto_weights"
        ), patch("trl.KTOTrainer", return_value=_mock_pref_instance()) as m_kto, patch(
            "trl.SFTTrainer"
        ) as m_sft, patch("trl.KTOConfig"):
            run = trainer.train("dummy", steps=5)

        assert m_kto.called, "KTOTrainer should be constructed for method='kto'."
        assert not m_sft.called
        assert run is not None

    def test_kto_passes_no_ref_model(self, temp_dir):
        """KTOTrainer is built WITHOUT an explicit ref_model (frozen base is ref)."""
        trainer = _pref_trainer_ready(temp_dir, "kto")
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch.object(
            trainer, "_auto_balance_kto_weights"
        ), patch("trl.KTOTrainer", return_value=_mock_pref_instance()) as m_kto, patch(
            "trl.SFTTrainer"
        ), patch("trl.KTOConfig"):
            trainer.train("dummy", steps=5)

        _, call_kwargs = m_kto.call_args
        assert "ref_model" not in call_kwargs, (
            "KTOTrainer must NOT receive an explicit ref_model — passing one "
            "loads a SECOND model and breaks the 16GB envelope. With the LoRA "
            "adapter attached, KTOTrainer's adapter-disable gives the reference "
            "for free."
        )

    def test_records_method_and_knobs_in_run_history(self, temp_dir):
        from backpropagate.checkpoints import RunHistoryManager

        trainer = _pref_trainer_ready(temp_dir, "kto", kto_beta=0.2)
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch.object(
            trainer, "_auto_balance_kto_weights"
        ), patch("trl.KTOTrainer", return_value=_mock_pref_instance()), patch(
            "trl.SFTTrainer"
        ), patch("trl.KTOConfig"):
            run = trainer.train("dummy", steps=5)

        hp = RunHistoryManager(str(temp_dir)).get_run(run.run_id).get(
            "hyperparameters", {}
        )
        assert hp.get("method") == "kto"
        assert hp.get("kto_beta") == pytest.approx(0.2)

    def test_oom_retry_rebuilds_kto_trainer(self, temp_dir):
        trainer = _pref_trainer_ready(temp_dir, "kto")
        trainer.oom_recovery = True
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=4)
        script = _OOMScript(oom_count=1)

        with patch.object(trainer, "_load_dataset", return_value=mock_ds), patch.object(
            trainer, "_auto_balance_kto_weights"
        ), patch("trl.KTOTrainer", side_effect=script.factory), patch(
            "trl.SFTTrainer"
        ) as m_sft, patch("trl.KTOConfig"):
            trainer.train("dummy", steps=5)

        assert script.construct_calls >= 2
        assert not m_sft.called


# =============================================================================
# KTO auto-weighting
# =============================================================================


class TestKtoAutoWeighting:
    """v1.6 C2: _auto_balance_kto_weights rebalances toward [1:1, 4:3]."""

    def _ds(self, n_pos: int, n_neg: int):
        from datasets import Dataset

        rows = [{"prompt": "p", "completion": "c", "label": True}] * n_pos
        rows += [{"prompt": "p", "completion": "c", "label": False}] * n_neg
        return Dataset.from_list(rows)

    def test_balanced_keeps_seed_weights(self):
        trainer = _trainer_for_kto_weights()
        trainer._auto_balance_kto_weights(self._ds(5, 5))
        assert trainer._kto_resolved_desirable_weight == pytest.approx(1.0)
        assert trainer._kto_resolved_undesirable_weight == pytest.approx(1.0)

    def test_pos_heavy_scales_undesirable_up_to_upper_edge(self):
        """8:2 (seed ratio 4.0) → undesirable scaled so effective lands at 4/3."""
        trainer = _trainer_for_kto_weights()
        trainer._auto_balance_kto_weights(self._ds(8, 2))
        d = trainer._kto_resolved_desirable_weight
        u = trainer._kto_resolved_undesirable_weight
        eff = (d * 8) / (u * 2)
        assert eff == pytest.approx(4.0 / 3.0, rel=1e-4)

    def test_neg_heavy_scales_desirable_up_to_lower_edge(self):
        """2:8 (seed ratio 0.25) → desirable scaled so effective lands at 1.0."""
        trainer = _trainer_for_kto_weights()
        trainer._auto_balance_kto_weights(self._ds(2, 8))
        d = trainer._kto_resolved_desirable_weight
        u = trainer._kto_resolved_undesirable_weight
        eff = (d * 2) / (u * 8)
        assert eff == pytest.approx(1.0, rel=1e-4)

    def test_one_sided_keeps_seed_and_does_not_crash(self):
        trainer = _trainer_for_kto_weights()
        trainer._auto_balance_kto_weights(self._ds(5, 0))
        assert trainer._kto_resolved_desirable_weight == pytest.approx(1.0)
        assert trainer._kto_resolved_undesirable_weight == pytest.approx(1.0)

    def test_unreadable_dataset_degrades_to_seed(self):
        """A dataset with no 'label' column leaves seed weights in place."""
        trainer = _trainer_for_kto_weights()
        bad = MagicMock()
        bad.__getitem__ = MagicMock(side_effect=KeyError("label"))
        trainer._auto_balance_kto_weights(bad)
        assert trainer._kto_resolved_desirable_weight == pytest.approx(1.0)
        assert trainer._kto_resolved_undesirable_weight == pytest.approx(1.0)


def _trainer_for_kto_weights():
    from backpropagate.trainer import Trainer

    with patch("torch.cuda.is_available", return_value=False):
        return Trainer(method="kto", mode="lora", use_unsloth=False, backend="cuda")


# =============================================================================
# _load_dataset routing — paired (simpo) vs unpaired (kto)
# =============================================================================


class TestLoadDatasetRouting:
    """v1.6 C2: SimPO uses the paired path; KTO uses the unpaired path."""

    def test_simpo_uses_preference_path(self, temp_dir):
        """method='simpo' calls to_preference_dataset (paired, like ORPO)."""
        from backpropagate.datasets import DatasetLoader
        from backpropagate.trainer import Trainer

        # spec=DatasetLoader so the isinstance(dataset, DatasetLoader) branch in
        # _load_dataset is taken (a bare MagicMock fails that check).
        loader = MagicMock(spec=DatasetLoader)
        loader.validation_result = MagicMock(
            warnings=[], is_valid=True, errors=[], error_count=0, error_rate=0.0
        )
        loader.detected_format = MagicMock()
        loader.to_preference_dataset.return_value = _len_ds(4)

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="simpo", mode="lora", use_unsloth=False, backend="cuda",
                output_dir=str(temp_dir),
            )
        trainer._load_dataset(loader, method="simpo")
        assert loader.to_preference_dataset.called
        assert not loader.to_kto_dataset.called

    def test_kto_uses_kto_path(self, temp_dir):
        """method='kto' calls to_kto_dataset (unpaired)."""
        from backpropagate.datasets import DatasetLoader
        from backpropagate.trainer import Trainer

        loader = MagicMock(spec=DatasetLoader)
        loader.validation_result = MagicMock(
            warnings=[], is_valid=True, errors=[], error_count=0, error_rate=0.0
        )
        loader.detected_format = MagicMock()
        loader.to_kto_dataset.return_value = _len_ds(4)

        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="kto", mode="lora", use_unsloth=False, backend="cuda",
                output_dir=str(temp_dir),
            )
        trainer._load_dataset(loader, method="kto")
        assert loader.to_kto_dataset.called
        assert not loader.to_preference_dataset.called

    def test_simpo_in_memory_missing_columns_raises(self, temp_dir):
        """An in-memory Dataset without chosen/rejected → DatasetFormatError."""
        from datasets import Dataset

        from backpropagate.exceptions import DatasetFormatError
        from backpropagate.trainer import Trainer

        ds = Dataset.from_list([{"text": "x"}])
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="simpo", mode="lora", use_unsloth=False, backend="cuda",
                output_dir=str(temp_dir),
            )
        with pytest.raises(DatasetFormatError):
            trainer._load_dataset(ds, method="simpo")

    def test_kto_in_memory_missing_columns_raises(self, temp_dir):
        """An in-memory Dataset without completion/label → DatasetFormatError."""
        from datasets import Dataset

        from backpropagate.exceptions import DatasetFormatError
        from backpropagate.trainer import Trainer

        ds = Dataset.from_list([{"chosen": "a", "rejected": "b"}])
        with patch("torch.cuda.is_available", return_value=False):
            trainer = Trainer(
                method="kto", mode="lora", use_unsloth=False, backend="cuda",
                output_dir=str(temp_dir),
            )
        with pytest.raises(DatasetFormatError):
            trainer._load_dataset(ds, method="kto")


def _len_ds(n: int):
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=n)
    return ds
