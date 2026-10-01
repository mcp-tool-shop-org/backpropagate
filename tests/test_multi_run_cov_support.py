"""Shared helpers for the ``tests/test_multi_run_cov_*.py`` coverage suites.

Nothing in here is collected as a test. It builds the pieces that let
``MultiRunTrainer`` run its real orchestration code on CPU:

* a REAL tiny PEFT/LoRA adapter on a REAL tiny Llama (``tests/helpers/tiny_models``),
  so adapter extraction, SLAO merging and weight loading are never mocked;
* a REAL tiny word-level tokenizer and REAL ``datasets.Dataset`` rows;
* :func:`make_fake_sft` -- the ONLY stand-in for the heavy boundary: the inner
  per-run ``trl.SFTTrainer``. It records what ``_execute_run`` passed to it
  (constructor kwargs, the real ``SFTConfig``, callbacks) and "trains" by
  writing deterministic values into the real adapter, so merges can be checked
  against hand-computed numbers;
* :class:`FakeInnerTrainer` -- stands in for ``backpropagate.trainer.Trainer``
  (model loading is a GPU/HF-Hub boundary); it carries the real PEFT model and
  tokenizer and performs a real ``save_pretrained`` for ``save()``.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("trl")

from backpropagate.checkpoints import CheckpointManager, CheckpointPolicy, RunHistoryManager
from backpropagate.gpu_safety import GPUCondition, GPUStatus
from backpropagate.multi_run import MergeMode, MultiRunConfig, MultiRunTrainer
from backpropagate.slao import MergeStrategyConfig, SLAOConfig, SLAOMerger
from tests.helpers.tiny_models import sentences, tiny_llama, tiny_tokenizer

# Number of LoRA tensors in the tiny adapter: 2 layers x (q_proj, v_proj) x (A, B).
N_LAYERS = 2
N_A = N_LAYERS * 2
N_B = N_LAYERS * 2


def build_peft_llama(*, layers: int = N_LAYERS, r: int = 4):
    """A real PeftModel (LoRA on q_proj/v_proj) over a real tiny Llama, CPU only."""
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        tiny_llama(layers=layers),
        LoraConfig(
            r=r,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )


def lora_params(model) -> dict[str, torch.Tensor]:
    """Snapshot (clone) of every ``lora_`` parameter of ``model``."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def fill_adapter(model, *, run: int, b_value: float | None = None) -> None:
    """Write a deterministic "trained" adapter into ``model``.

    A matrices: seeded random (distinct per ``run``, full rank so the SLAO
    orthogonal init is well-defined). B matrices: the constant ``b_value``
    (default ``float(run)``) so EMA merges are hand-computable.
    """
    b_value = float(run) if b_value is None else b_value
    gen = torch.Generator().manual_seed(1000 + run)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_A." in name:
                p.copy_(torch.randn(p.shape, generator=gen))
            elif ".lora_B." in name:
                p.fill_(b_value)


class FakeInnerTrainer:
    """Stands in for ``backpropagate.trainer.Trainer`` after ``load_model()``."""

    def __init__(self, model=None, *, batch_size: int = 2, grad_accum: int = 1,
                 report_to: str = "none", **ctor_kwargs: Any) -> None:
        self._model = model if model is not None else build_peft_llama()
        self._tokenizer = tiny_tokenizer()
        self.ctor_kwargs = ctor_kwargs
        self.batch_size = batch_size
        self.gradient_accumulation = grad_accum
        self.max_seq_length = 32
        self.packing = False
        self.optim = None
        self.mode = "lora"
        self._train_on_responses = False
        self.use_unsloth = False
        self._response_markers_override = None
        self._report_to = report_to
        self.save_calls: list[tuple[str, dict[str, Any]]] = []
        self.pre_tokenize_calls: list[Any] = []
        self.fail_save_with: Exception | None = None
        self.load_model_calls = 0

    def load_model(self) -> None:
        self.load_model_calls += 1

    def _resolve_report_to(self) -> str:
        return self._report_to

    def _pre_tokenize(self, dataset):
        self.pre_tokenize_calls.append(dataset)
        return dataset

    def save(self, path: str, **kwargs: Any) -> None:
        self.save_calls.append((path, kwargs))
        if self.fail_save_with is not None:
            raise self.fail_save_with
        # REAL adapter write (adapter_config.json + adapter_model.safetensors).
        self._model.save_pretrained(path)


def make_fake_sft(train_fns: list | None = None, *, default_loss: float = 1.5):
    """Build a fresh stand-in ``SFTTrainer`` class.

    ``train_fns`` is consumed one entry per ``train()`` call; each entry is a
    callable ``fn(sft) -> result`` (it may raise). Once exhausted, ``train()``
    falls back to :func:`fill_adapter` with ``run`` = the call number and a
    small log history. Every instance is appended to ``cls.created`` and
    snapshots the adapter at the START of ``train()`` into ``start_params``.
    """

    class _FakeSFT:
        created: list[_FakeSFT] = []  # noqa: RUF012
        train_calls = 0
        _fns = list(train_fns or [])

        def __init__(self, model=None, processing_class=None, train_dataset=None,
                     args=None, callbacks=None, **kwargs: Any) -> None:
            self.model = model
            self.processing_class = processing_class
            self.train_dataset = train_dataset
            self.args = args
            self.callbacks = callbacks
            self.extra_kwargs = kwargs
            self.state = types.SimpleNamespace(log_history=[], global_step=0)
            self.start_params: dict[str, torch.Tensor] = {}
            type(self).created.append(self)

        def train(self):
            cls = type(self)
            cls.train_calls += 1
            call_no = cls.train_calls
            self.start_params = lora_params(self.model)
            if cls._fns:
                return cls._fns.pop(0)(self)
            fill_adapter(self.model, run=call_no)
            self.state.log_history = [
                {"loss": default_loss + 1.0, "step": 1},
                {"eval_loss": 9.9, "step": 2},  # must be ignored by loss extraction
                {"loss": default_loss, "step": 3},
            ]
            return types.SimpleNamespace(training_loss=default_loss)

    return _FakeSFT


def install_fake_sft(monkeypatch, fake_cls) -> None:
    """Make ``from trl import SFTTrainer`` (inside ``_execute_run``) yield ``fake_cls``."""
    import trl

    monkeypatch.setattr(trl, "SFTTrainer", fake_cls)


def text_dataset(n: int = 40, seed: int = 0):
    """A real ``datasets.Dataset`` of ChatML-ish ``text`` rows from the tiny vocab."""
    from datasets import Dataset

    return Dataset.from_dict({"text": sentences(n, seed=seed)})


def safe_gpu_status() -> GPUStatus:
    """A healthy GPU snapshot (the CUDA boundary is the only thing mocked)."""
    return GPUStatus(
        available=True,
        device_name="Fake GPU",
        vram_total_gb=24.0,
        condition=GPUCondition.SAFE,
        condition_reason="ok",
    )


def make_trainer(tmp_path: Path, *, inner: FakeInnerTrainer | None = None,
                 merge_mode: MergeMode = MergeMode.SLAO, **cfg: Any) -> MultiRunTrainer:
    """A ``MultiRunTrainer`` bootstrapped the way ``run()`` leaves it before the loop.

    Wires a real ``CheckpointManager``, ``RunHistoryManager`` and (in SLAO mode) a
    real ``SLAOMerger`` built from the same config fields ``run()`` uses, plus
    the fake inner trainer, so ``_execute_run`` can be called directly.
    """
    defaults: dict[str, Any] = {
        "num_runs": 2,
        "steps_per_run": 3,
        "samples_per_run": 8,
        "merge_mode": merge_mode,
        "checkpoint_dir": str(tmp_path),
        "enable_gpu_monitoring": False,
        "pause_on_overheat": False,
        "validate_every_run": False,
        "shuffle_data": False,
    }
    defaults.update(cfg)
    mrt = MultiRunTrainer(model="tiny-test", config=MultiRunConfig(**defaults))
    mrt._run_id = "abcdef0123456789"
    mrt._trainer = inner if inner is not None else FakeInnerTrainer()
    mrt._checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(tmp_path), policy=CheckpointPolicy()
    )
    mrt._run_history = RunHistoryManager(str(tmp_path))
    if merge_mode == MergeMode.SLAO:
        mrt._slao_merger = SLAOMerger(
            SLAOConfig(
                scaling_type="sqrt",
                use_orthogonal_init=True,
                use_adaptive_scaling=mrt.config.adaptive_scaling,
                use_layer_scaling=mrt.config.layer_scaling,
            ),
            strategy_config=MergeStrategyConfig(
                strategy=mrt.config.merge_strategy,
                trim_threshold=mrt.config.ties_trim_threshold,
                drop_rate=mrt.config.dare_drop_rate,
                dare_seed=mrt.config.dare_seed,
            ),
        )
    return mrt
