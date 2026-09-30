"""Engine B inside the real TRL ``SFTTrainer`` loop (CPU, tiny random models).

The engine's claim is that it is only an optimizer + callback, so the stock
trainer machinery keeps working: gradient accumulation, global grad-norm
clipping, LR scheduling, intermediate checkpoints and resume. These tests run
that machinery for real (bf16 model, bf16 autocast on CPU, gradient
checkpointing on) and check the engine's schedule against it. The library's
``Trainer`` wiring (gates, the one construction path, the end-to-end CPU run)
is tested at the bottom.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")

from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from backpropagate import block_engine as be  # noqa: E402
from backpropagate.exceptions import InvalidSettingError  # noqa: E402
from tests.helpers.tiny_models import sentences, tiny_llama, tiny_tokenizer  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _config(tmp_path, **over):
    kw = {
        "output_dir": str(tmp_path / "out"),
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": 6,
        "learning_rate": 5e-3,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "bf16": True,
        "use_cpu": True,
        "seed": 11,
        "data_seed": 11,
        "max_length": 24,
        "packing": False,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": 1.0,
    }
    kw.update(over)
    return SFTConfig(**kw)


def _dataset(n=48):
    return datasets.Dataset.from_dict({"text": sentences(n)})


class _Recorder(TrainerCallback):
    """Records the engine's view at every optimizer step and every micro-step."""

    def __init__(self, opt, model):
        self.opt, self.model = opt, model
        self.steps: list[dict] = []
        self.micro = 0
        self.prev = None
        self.grad_norms: list[float] = []

    def on_substep_end(self, args, state, control, **kw):
        self.micro += 1

    def on_pre_optimizer_step(self, args, state, control, **kw):
        # clip_grad_norm_ already ran: the norm over every grad is the post-clip norm
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.grad_norms.append(float(torch.norm(torch.stack([g.float().norm() for g in grads]))))
        self.active_during = self.opt.active_block_index
        self.snap = {n: p.detach().float().clone() for n, p in self.model.named_parameters()}

    def on_step_end(self, args, state, control, **kw):
        self.micro += 1
        changed = {
            n for n, p in self.model.named_parameters() if not torch.equal(p.detach().float(), self.snap[n])
        }
        self.steps.append(
            {
                "hf_step": state.global_step,
                "engine_step": self.opt.global_step,
                "active_during": self.active_during,
                "active_after": self.opt.active_block_index,
                "changed": changed,
            }
        )


def _run(tmp_path, model, *, k=2, ga=1, steps=6, order="ascending", writeback="stochastic", **cfg):
    args = _config(tmp_path, gradient_accumulation_steps=ga, max_steps=steps, **cfg)
    opt, cb = be.build_for_sft_trainer(
        model, args, switch_block_every=k, block_order=order, block_writeback=writeback
    )
    rec = _Recorder(opt, model)
    tr = SFTTrainer(
        model=model, processing_class=tiny_tokenizer(), train_dataset=_dataset(),
        args=args, callbacks=[cb, rec], optimizers=(opt, None),
    )
    tr.train()
    return tr, opt, rec


class TestInsideSFTTrainer:
    @pytest.mark.parametrize("ga", [1, 4])
    def test_switches_on_optimizer_steps_under_accumulation(self, tmp_path, ga):
        model = tiny_llama(layers=3, dtype=torch.bfloat16)
        _, opt, rec = _run(tmp_path, model, k=2, ga=ga, steps=6)
        assert [s["hf_step"] for s in rec.steps] == list(range(1, 7))
        assert [s["engine_step"] for s in rec.steps] == list(range(1, 7))
        # ascending, K=2: blocks 0,0,1,1,2,2 -> switch after steps 2, 4, 6
        assert [s["active_during"] for s in rec.steps] == [0, 0, 1, 1, 2, 2]
        assert [e["at_global_step"] for e in opt.switch_log[:3]] == [2, 4, 6]
        assert rec.micro == 6 * ga  # every micro-batch ran; one optimizer step per GA cycle

    def test_only_the_active_block_changes_each_step(self, tmp_path):
        model = tiny_llama(layers=3, dtype=torch.bfloat16)
        _, opt, rec = _run(tmp_path, model, k=2, steps=6, order="random")
        for s in rec.steps:
            allowed = set(opt.partition.blocks[s["active_during"]].param_names)
            assert s["changed"], s
            assert s["changed"] <= allowed, s

    def test_model_is_plain_bf16_after_training(self, tmp_path):
        model = tiny_llama(layers=2, dtype=torch.bfloat16)
        _run(tmp_path, model, k=4, steps=3)
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        assert all(p.requires_grad for p in model.parameters())

    def test_global_grad_norm_clipping_applies(self, tmp_path):
        model = tiny_llama(layers=2, dtype=torch.bfloat16)
        tr, _, rec = _run(tmp_path, model, k=3, steps=4, max_grad_norm=1e-3)
        logged = [h["grad_norm"] for h in tr.state.log_history if "grad_norm" in h]
        assert logged and all(float(g) > 1e-3 for g in logged)  # pre-clip norm
        assert all(n <= 1e-3 * 1.01 for n in rec.grad_norms)  # post-clip norm

    def test_input_grad_hook_removed_so_backward_stops(self, tmp_path):
        model = tiny_llama(layers=3, dtype=torch.bfloat16)
        seen = []
        model.model.layers[0].register_forward_hook(
            lambda _m, _i, o: seen.append((o[0] if isinstance(o, tuple) else o).requires_grad)
        )
        # descending: the head block, then layer 2, then layer 1 — never layer 0
        _run(tmp_path, model, k=1, steps=3, order="descending")
        assert getattr(model, "_require_grads_hook", None) is None
        assert seen and not any(seen)

    def test_lr_scheduler_drives_the_engine(self, tmp_path):
        model = tiny_llama(layers=2, dtype=torch.bfloat16)
        tr, opt, _ = _run(tmp_path, model, k=2, steps=4)
        lrs = [h["learning_rate"] for h in tr.state.log_history if "learning_rate" in h]
        assert lrs[0] > lrs[-1]  # linear decay reached our param groups
        assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)

    def test_16bit_model_without_autocast_is_refused(self, tmp_path):
        model = tiny_llama(layers=2, dtype=torch.bfloat16)
        with pytest.raises(InvalidSettingError):
            be.build_for_sft_trainer(model, _config(tmp_path, bf16=False))


class TestCheckpointResume:
    @pytest.mark.parametrize("save_at", [3, 4])
    def test_resume_matches_straight_run(self, tmp_path, save_at):
        """2K straight == K (+1), checkpoint, resume the rest. K=3; save_at=3
        is a block boundary, save_at=4 is mid-block (master + Adam moments in
        the checkpoint). Round-to-nearest write-back, random order."""
        base = tiny_llama(layers=3, dtype=torch.bfloat16, seed=2)
        straight = copy.deepcopy(base)
        tr_a, opt_a, _ = _run(
            tmp_path / "a", straight, k=3, steps=6, order="random", writeback="nearest",
            save_strategy="steps", save_steps=save_at,
        )
        ckpt = Path(tmp_path / "a" / "out" / f"checkpoint-{save_at}")
        assert (ckpt / "optimizer.pt").exists()
        saved = torch.load(ckpt / "optimizer.pt", weights_only=True)
        assert saved["block_engine"]["global_step"] == save_at
        assert saved["block_engine"]["steps_in_block"] == save_at % 3

        resumed = copy.deepcopy(base)
        args = _config(tmp_path / "b", max_steps=6)
        opt_b, cb = be.build_for_sft_trainer(
            resumed, args, switch_block_every=3, block_order="random", block_writeback="nearest"
        )
        tr_b = SFTTrainer(
            model=resumed, processing_class=tiny_tokenizer(), train_dataset=_dataset(),
            args=args, callbacks=[cb], optimizers=(opt_b, None),
        )
        tr_b.train(resume_from_checkpoint=str(ckpt))
        assert opt_b.global_step == 6
        assert [e["block"] for e in opt_b.switch_log][-1] == [e["block"] for e in opt_a.switch_log][-1]
        for (n, a), (_, b) in zip(straight.named_parameters(), resumed.named_parameters()):
            assert torch.equal(a, b), n

    def test_checkpoint_weights_hold_the_fp32_master(self, tmp_path):
        """Documented behaviour: an intermediate checkpoint's model file holds
        the active block in fp32 (the exact master); every other tensor is
        bf16. The final save after training is all-bf16."""
        from safetensors.torch import load_file

        model = tiny_llama(layers=2, dtype=torch.bfloat16)
        _, opt, _ = _run(tmp_path, model, k=4, steps=2, save_strategy="steps", save_steps=1)
        sd = load_file(str(Path(tmp_path / "out" / "checkpoint-1" / "model.safetensors")))
        fp32 = {n for n, t in sd.items() if t.dtype == torch.float32}
        assert fp32 and fp32 <= set(opt.partition.blocks[opt.switch_log[0]["index"]].param_names)


# ---------------------------------------------------------------------------
# Library Trainer wiring
# ---------------------------------------------------------------------------
class TestLibraryTrainerWiring:
    def _trainer(self, **kw):
        from backpropagate.trainer import Trainer

        base = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "mode": "full", "use_unsloth": False,
                    "full_ft_engine": "block"}
        base.update(kw)
        return Trainer(**base)

    @pytest.mark.parametrize("method", ["orpo", "simpo", "kto"])
    def test_preference_methods_refused(self, method):
        with pytest.raises(InvalidSettingError) as ei:
            self._trainer(method=method)
        assert ei.value.code == "CONFIG_INVALID_SETTING"
        assert "full_ft_engine" in str(ei.value)

    def test_offload_combination_refused(self):
        with pytest.raises(InvalidSettingError) as ei:
            self._trainer(full_ft_offload=True)
        assert ei.value.code == "CONFIG_INVALID_SETTING"
        assert "full_ft_offload" in str(ei.value)

    def test_lora_mode_refused(self):
        with pytest.raises(InvalidSettingError):
            self._trainer(mode="lora")

    @pytest.mark.parametrize("kw", [{"full_ft_engine": "swap"}, {"switch_block_every": 0},
                                    {"block_order": "zigzag"}, {"block_writeback": "floor"}])
    def test_bad_values_refused(self, kw):
        with pytest.raises(InvalidSettingError):
            self._trainer(**kw)

    def test_defaults_resolved(self):
        t = self._trainer()
        assert (t.full_ft_engine, t.switch_block_every, t.block_order, t.block_writeback,
                t.block_train_embeddings) == ("block", 50, "random", "stochastic", True)
        assert t.use_unsloth is False

    def test_default_engine_leaves_everything_off(self):
        from backpropagate.trainer import Trainer

        t = Trainer(model="Qwen/Qwen2.5-1.5B-Instruct", mode="full", use_unsloth=False)
        assert t.full_ft_engine == "default"
        assert t._block_engine is None

    def test_block_ceiling_admits_7b_on_32gb(self, monkeypatch):
        import backpropagate.trainer as tmod

        monkeypatch.setattr(tmod, "_detect_total_vram_gb", lambda: 31.8)
        t = self._trainer(model="Qwen/Qwen2.5-7B-Instruct")
        assert t._resolve_full_ft_ceilings()[0] == 8.0
        from backpropagate.exceptions import FullFinetuneModelTooLargeError

        with pytest.raises(FullFinetuneModelTooLargeError):
            self._trainer(model="Qwen/Qwen2.5-7B-Instruct", full_ft_engine=None)

    def test_ceiling_tiers(self):
        import backpropagate.trainer as tmod

        f = tmod._full_ft_block_ceiling_for_vram
        assert [f(v) for v in (None, 16, 23.6, 31.8, 48)] == [4.0, 4.0, 6.0, 8.0, 12.0]
        assert [f(v, train_embeddings=False) for v in (16, 24, 32, 48)] == [5.0, 8.0, 11.0, 16.0]

    def test_build_trainer_default_path_unchanged(self, tmp_path):
        """Default engine: SFTTrainer gets exactly the pre-engine kwargs."""
        from backpropagate.trainer import Trainer

        t = Trainer(model="Qwen/Qwen2.5-1.5B-Instruct", mode="full", use_unsloth=False)
        t._model, t._tokenizer = object(), object()
        with patch("trl.SFTTrainer") as sft:
            t._build_trainer(_config(tmp_path), [], None)
        assert set(sft.call_args.kwargs) == {
            "model", "processing_class", "train_dataset", "args", "callbacks"
        }

    def test_build_trainer_block_path_and_oom_rebuild(self, tmp_path):
        t = self._trainer(switch_block_every=3)
        t._model, t._tokenizer = tiny_llama(layers=2, dtype=torch.bfloat16), object()
        with patch("trl.SFTTrainer") as sft:
            t._build_trainer(_config(tmp_path), [], None)
            first = t._block_engine
            opt, _sched = sft.call_args.kwargs["optimizers"]
            assert opt is first and _sched is None
            assert any(type(c).__name__ == "BlockEngineCallback" for c in sft.call_args.kwargs["callbacks"])
            assert any(p.dtype == torch.float32 for p in t._model.parameters())
            # OOM retry: the same helper rebuilds; the old engine is finalized first
            t._build_trainer(_config(tmp_path), [], None)
        assert t._block_engine is not first
        assert first._finalized

    def test_training_args_optim_is_named(self, tmp_path):
        t = self._trainer()
        t.output_dir = tmp_path
        args = t._build_training_args(steps=2, report_to="none", run_name=None)
        assert getattr(args.optim, "value", args.optim) == "adamw_torch"
        assert args.gradient_checkpointing is True


class TestLibraryEndToEnd:
    def test_trainer_train_runs_block_engine_on_cpu(self, tmp_path, monkeypatch):
        """Trainer(mode='full', full_ft_engine='block').train() on a tiny
        local model directory. On CPU the library loads fp32 (no autocast), so
        this checks the wiring end to end, not the bf16 precision path (the
        classes above do that)."""
        monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")
        from backpropagate.trainer import Trainer

        model_dir = tmp_path / "tiny"
        tiny_llama(layers=2).save_pretrained(model_dir)
        tiny_tokenizer().save_pretrained(model_dir)
        data = tmp_path / "d.jsonl"
        with open(data, "w") as fh:
            for s in sentences(16):
                fh.write(json.dumps({"messages": [{"role": "user", "content": s},
                                                  {"role": "assistant", "content": s}]}) + "\n")
        t = Trainer(model=str(model_dir), mode="full", use_unsloth=False, full_ft_engine="block",
                    switch_block_every=2, block_order="ascending", batch_size=2, max_seq_length=32,
                    output_dir=str(tmp_path / "out"), learning_rate=1e-3, packing=False,
                    report_to="none")
        run = t.train(str(data), steps=4)
        meta = run.metadata["block_engine"]
        assert meta["optimizer_steps"] == 4
        assert [s["at_global_step"] for s in meta["switches"]][:2] == [2, 4]
        assert all(p.requires_grad for p in t._model.parameters())
        assert os.path.isdir(tmp_path / "out")


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
class TestCli:
    def test_flag_defaults_and_choices_mirror_the_engine(self):
        from backpropagate.cli import create_parser

        p = create_parser()
        a = p.parse_args(["train", "-d", "x.jsonl"])
        assert (a.full_ft_engine, a.switch_block_every, a.block_order, a.block_writeback,
                a.block_freeze_embeddings) == ("default", be.DEFAULT_SWITCH_BLOCK_EVERY,
                                               be.DEFAULT_BLOCK_ORDER, be.DEFAULT_BLOCK_WRITEBACK, False)
        import argparse

        sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        train = sub.choices["train"]
        choices = {o.dest: tuple(o.choices) for o in train._actions if o.choices}
        assert choices["full_ft_engine"] == be.FULL_FT_ENGINES
        assert choices["block_order"] == be.BLOCK_ORDERS
        assert choices["block_writeback"] == be.BLOCK_WRITEBACK_MODES
        with pytest.raises(SystemExit):
            p.parse_args(["train", "-d", "x.jsonl", "--switch-block-every", "0"])

    def _forwarded(self, tmp_path, extra):
        from unittest.mock import MagicMock

        from backpropagate.cli import cmd_train, create_parser

        args = create_parser().parse_args(
            ["train", "-d", "x.jsonl", "--mode", "full", "-o", str(tmp_path), *extra]
        )
        result = MagicMock(final_loss=0.1, duration_seconds=1.0)
        trainer = MagicMock()
        trainer.train.return_value = result
        trainer.save.return_value = str(tmp_path)
        with patch("backpropagate.trainer.Trainer", return_value=trainer) as cls, \
             patch("backpropagate.trainer.TrainingCallback"):
            cmd_train(args)
        return cls.call_args.kwargs

    def test_default_run_forwards_nothing_new(self, tmp_path):
        kw = self._forwarded(tmp_path, [])
        assert not ({"full_ft_engine", "switch_block_every", "block_order", "block_writeback",
                     "block_train_embeddings"} & set(kw))

    def test_block_run_forwards_the_knobs(self, tmp_path):
        kw = self._forwarded(tmp_path, ["--full-ft-engine", "block", "--switch-block-every", "10",
                                        "--block-order", "descending", "--block-writeback", "nearest",
                                        "--block-freeze-embeddings"])
        assert kw["full_ft_engine"] == "block"
        assert kw["switch_block_every"] == 10
        assert kw["block_order"] == "descending"
        assert kw["block_writeback"] == "nearest"
        assert kw["block_train_embeddings"] is False
