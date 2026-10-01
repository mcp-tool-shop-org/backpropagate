"""Coverage tests for ``backpropagate.eval`` against a REAL tiny model.

``evaluate_run`` is driven end to end: a run is recorded in a real
``RunHistoryManager`` file, a real tiny Llama + LoRA adapter (random weights,
saved to ``tmp_path``) is loaded on CPU, and held-out loss, perplexity,
generations and task metrics are computed and persisted back to the history.

A scripted stub model (real torch tensors, real tokenizer) is used only where
the real model cannot be made to produce a specific condition (NaN loss, an
astronomically high loss, a generation crash, a known completion to score
against). It is patched in at the model-loading seam
``eval._load_model_and_tokenizer``.

Nothing touches a GPU or the network: the base model is a local directory.
"""

from __future__ import annotations

import json
import logging
import math
import types
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model

import backpropagate.eval as ev
from backpropagate.checkpoints import RunHistoryManager
from backpropagate.exceptions import TrainingError, UserInputError
from tests.helpers.tiny_models import tiny_llama, tiny_tokenizer

LOGGER = "backpropagate.eval"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def assets(tmp_path_factory) -> types.SimpleNamespace:
    root = tmp_path_factory.mktemp("eval_assets")
    base = root / "base"
    tiny_llama(layers=2).save_pretrained(base)
    tiny_tokenizer().save_pretrained(base)

    torch.manual_seed(7)
    peft_model = get_peft_model(
        tiny_llama(layers=2),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    )
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(0.0, 1.0)
    adapter = root / "adapter"
    peft_model.save_pretrained(adapter)
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = str(base)
    (adapter / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return types.SimpleNamespace(base=base, adapter=adapter)


HELDOUT_ROWS = [
    {"instruction": "what is two plus four", "output": "the cat sat on the mat"},
    {"instruction": "a dog ran to the park", "output": "yes and no"},
    {"instruction": "user the cat", "output": "assistant the dog ran"},
]


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _make_run(
    tmp_path: Path,
    assets,
    *,
    run_id: str = "run-abcdef123456",
    adapter: bool = True,
    hyperparameters: dict | None = None,
    dataset_info=None,
) -> tuple[str, str]:
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    RunHistoryManager(str(out)).record_run_started(
        run_id=run_id,
        model_name=str(assets.base),
        dataset_info=dataset_info,
        hyperparameters=hyperparameters or {},
        checkpoint_path=str(assets.adapter) if adapter else None,
    )
    return str(out), run_id


# =============================================================================
# evaluate_run: real model, end to end
# =============================================================================


class TestEvaluateRunRealModel:
    def test_loss_perplexity_generations_and_persistence(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets)
        heldout = _write_jsonl(tmp_path / "held.jsonl", HELDOUT_ROWS)
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("what is two plus four\nthe cat sat\n\nignored third\n", encoding="utf-8")

        result = ev.evaluate_run(
            run_id, output_dir=out, heldout=str(heldout), prompts=str(prompts), n=2,
            max_new_tokens=3, temperature=0.7, seed=3,
        )

        assert result.run_id == run_id and result.model_name == str(assets.base)
        assert result.held_out_loss is not None and result.held_out_loss > 0
        assert result.perplexity == pytest.approx(math.exp(result.held_out_loss))
        assert result.n_prompts == 2
        assert [g.prompt for g in result.generations] == ["what is two plus four", "the cat sat"]
        assert all(isinstance(g.completion, str) for g in result.generations)
        assert result.task_metrics == {} and result.eval_n == 0 and result.metric_ci is None

        # persisted back onto the run-history entry
        stored = RunHistoryManager(out).get_run(run_id)
        assert stored["eval"]["held_out_loss"] == pytest.approx(result.held_out_loss)
        assert [g["prompt"] for g in stored["eval"]["generations"]] == [g.prompt for g in result.generations]

        # a second pass is byte-for-byte reproducible (fixed seed, eval mode)
        again = ev.evaluate_run(
            run_id, output_dir=out, heldout=str(heldout), prompts=str(prompts), n=2,
            max_new_tokens=3, temperature=0.7, seed=3,
        )
        assert again.held_out_loss == result.held_out_loss
        assert [g.completion for g in again.generations] == [g.completion for g in result.generations]

    def test_adapter_changes_the_loss_relative_to_the_bare_base(self, tmp_path, assets):
        heldout = _write_jsonl(tmp_path / "held.jsonl", HELDOUT_ROWS)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        out_a, rid_a = _make_run(tmp_path / "a", assets, adapter=True)
        out_b, rid_b = _make_run(tmp_path / "b", assets, adapter=False)
        with_adapter = ev.evaluate_run(rid_a, output_dir=out_a, heldout=str(heldout), n=1, max_new_tokens=1)
        bare = ev.evaluate_run(rid_b, output_dir=out_b, heldout=str(heldout), n=1, max_new_tokens=1)
        assert with_adapter.held_out_loss != pytest.approx(bare.held_out_loss, abs=1e-3)

    def test_base_only_run_warns_about_the_missing_adapter(self, tmp_path, assets, caplog):
        out, run_id = _make_run(tmp_path, assets, adapter=False)
        heldout = _write_jsonl(tmp_path / "held.jsonl", HELDOUT_ROWS)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = ev.evaluate_run(run_id, output_dir=out, heldout=str(heldout), n=1, max_new_tokens=1)
        assert result.held_out_loss is not None
        assert any("no adapter directory on disk" in r.getMessage() for r in caplog.records)

    def test_run_id_prefix_resolves_to_the_canonical_id(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets)
        heldout = _write_jsonl(tmp_path / "held.jsonl", HELDOUT_ROWS)
        result = ev.evaluate_run("run-abc", output_dir=out, heldout=str(heldout), n=1, max_new_tokens=1)
        assert result.run_id == run_id

    def test_max_seq_length_from_hyperparameters_truncates_scoring(self, tmp_path, assets):
        heldout = _write_jsonl(tmp_path / "held.jsonl", HELDOUT_ROWS)
        (tmp_path / "short").mkdir()
        (tmp_path / "long").mkdir()
        (tmp_path / "bad").mkdir()
        out_s, rid_s = _make_run(tmp_path / "short", assets, hyperparameters={"max_seq_length": 3})
        out_l, rid_l = _make_run(tmp_path / "long", assets, hyperparameters={})
        out_b, rid_b = _make_run(tmp_path / "bad", assets, hyperparameters={"max_seq_length": "not-a-number"})
        kwargs = {"heldout": str(heldout), "n": 1, "max_new_tokens": 1}
        short = ev.evaluate_run(rid_s, output_dir=out_s, **kwargs)
        long_ = ev.evaluate_run(rid_l, output_dir=out_l, **kwargs)
        bad = ev.evaluate_run(rid_b, output_dir=out_b, **kwargs)
        assert short.held_out_loss != pytest.approx(long_.held_out_loss, abs=1e-4)
        # an unparsable value falls back to the 1024 default instead of crashing
        assert bad.held_out_loss == pytest.approx(long_.held_out_loss)

    def test_reference_metrics_against_real_generations(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets)
        heldout = _write_jsonl(tmp_path / "held.jsonl", HELDOUT_ROWS)
        run = RunHistoryManager(out).get_run(run_id)
        model, tok = ev._load_model_and_tokenizer(run)
        prompts = ["what is two plus four", "the cat sat on"]
        gens = ev._generate(model, tok, prompts, max_new_tokens=4, temperature=0.0, seed=0)
        references = [{"prompt": g.prompt, "reference": g.completion} for g in gens]
        references[1] = {"prompt": gens[1].prompt, "references": ["totally different", gens[1].completion]}

        result = ev.evaluate_run(
            run_id, output_dir=out, heldout=str(heldout), n=1, max_new_tokens=4,
            references=references,
        )
        # references are the model's own greedy output -> perfect scores
        assert result.task_metrics == {"normalized_exact_match": 1.0, "token_f1": 1.0}
        assert result.eval_n == 2
        assert set(result.metric_ci) == {"normalized_exact_match", "token_f1"}
        assert all(0.0 <= hw <= 1.0 for hw in result.metric_ci.values())

        wrong = ev.evaluate_run(
            run_id, output_dir=out, heldout=str(heldout), n=1, max_new_tokens=4,
            references=[{"prompt": p, "reference": "zzz qqq unattainable"} for p in prompts],
            metrics=["normalized_exact_match", "contains"],
        )
        assert wrong.task_metrics == {"normalized_exact_match": 0.0, "contains": 0.0}
        assert wrong.to_dict()["task_metrics"] == wrong.task_metrics


class TestModelLoadFailures:
    """Mocks: ``AutoModelForCausalLM.from_pretrained`` (the hub / disk loading boundary)."""

    def test_trust_remote_code_refusal_becomes_the_structured_opt_in_error(self, tmp_path, assets, monkeypatch):
        from transformers import AutoModelForCausalLM

        from backpropagate.exceptions import TrustRemoteCodeRequiredError

        def refuse(*a, **k):
            raise ValueError("requires you to execute the modeling code: set trust_remote_code=True")

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", refuse)
        out, run_id = _make_run(tmp_path, assets)
        with pytest.raises(TrustRemoteCodeRequiredError) as exc:
            ev.evaluate_run(run_id, output_dir=out, heldout_texts=["the cat"], n=1)
        assert exc.value.code == "CONFIG_TRUST_REMOTE_CODE_REQUIRED"
        assert "BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE=true" in (exc.value.suggestion or "")

    def test_any_other_load_failure_names_the_run_model_and_checkpoint(self, tmp_path, assets, monkeypatch):
        from transformers import AutoModelForCausalLM

        def offline(*a, **k):
            raise OSError("cannot reach the hub")

        monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", offline)
        out, run_id = _make_run(tmp_path, assets)
        with pytest.raises(TrainingError) as exc:
            ev.evaluate_run(run_id, output_dir=out, heldout_texts=["the cat"], n=1)
        assert exc.value.code == "RUNTIME_EVAL_FAILED"
        # the message embeds repr()s, so compare against repr (backslashes double on Windows)
        assert f"run {run_id!r}" in exc.value.message
        assert f"model={str(assets.base)!r}" in exc.value.message
        assert f"checkpoint={str(assets.adapter)!r}" in exc.value.message
        assert "cannot reach the hub" in exc.value.message
        assert isinstance(exc.value.__cause__, OSError)


class TestHeldOutResolution:
    def test_unresolvable_without_heldout_or_dataset(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets, dataset_info="in-memory-object")
        with pytest.raises(UserInputError) as exc:
            ev.evaluate_run(run_id, output_dir=out)
        assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"
        assert "'in-memory-object'" in exc.value.message

    def test_resplit_of_recorded_dataset_warns_about_overlap(self, tmp_path, assets, caplog):
        ds = _write_jsonl(tmp_path / "train.jsonl", HELDOUT_ROWS * 4)  # 12 rows
        out, run_id = _make_run(tmp_path, assets, dataset_info=str(ds))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = ev.evaluate_run(run_id, output_dir=out, n=1, max_new_tokens=1)
        assert result.held_out_loss is not None
        assert any("OVERLAP WITH THE TRAINING SPLIT IS POSSIBLE" in r.getMessage() for r in caplog.records)

    def test_resplit_with_empty_test_slice_is_unresolved(self, tmp_path, assets, caplog):
        ds = tmp_path / "empty-dataset.jsonl"  # exists, but holds no rows to split
        ds.write_text("", encoding="utf-8")
        out, run_id = _make_run(tmp_path, assets, dataset_info=str(ds))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with pytest.raises(UserInputError) as exc:
                ev.evaluate_run(run_id, output_dir=out)
        assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"
        assert any("empty held-out slice" in r.getMessage() for r in caplog.records)

    def test_empty_heldout_file_is_unresolved(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(UserInputError) as exc:
            ev.evaluate_run(run_id, output_dir=out, heldout=str(empty))
        assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"
        assert "no usable samples" in exc.value.message

    def test_in_memory_texts_take_precedence_and_are_cleaned(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets)
        result = ev.evaluate_run(
            run_id, output_dir=out, heldout="/does/not/exist.jsonl",
            heldout_texts=["  the cat sat on the mat  ", "", "   "], n=1, max_new_tokens=1,
        )
        assert result.held_out_loss is not None  # the bogus path was never opened

    def test_all_blank_in_memory_texts_are_rejected(self, tmp_path, assets):
        out, run_id = _make_run(tmp_path, assets)
        with pytest.raises(UserInputError) as exc:
            ev.evaluate_run(run_id, output_dir=out, heldout_texts=["", "  "])
        assert exc.value.code == "INPUT_EVAL_HELDOUT_UNRESOLVED"

    def test_loader_to_texts_skips_blank_rows_and_coerces_non_dicts(self):
        class FakeLoader:
            def to_chatml(self):
                return [{"text": " kept "}, {"text": "   "}, "raw row", {"other": 1}]

        assert ev._loader_to_texts(FakeLoader()) == ["kept", "raw row"]

    def test_unknown_run(self, tmp_path):
        with pytest.raises(UserInputError) as exc:
            ev.evaluate_run("nope", output_dir=str(tmp_path))
        assert exc.value.code == "INPUT_EVAL_RUN_NOT_FOUND"
        assert "backprop runs" in (exc.value.suggestion or "")


class TestPromptLoading:
    def test_default_prompts_truncate_to_n(self):
        assert ev._load_prompts(None, 3) == ev.DEFAULT_PROMPTS[:3]
        assert ev._load_prompts(None, 0) == ev.DEFAULT_PROMPTS  # n <= 0 keeps all

    def test_file_formats(self, tmp_path):
        p = tmp_path / "p.jsonl"
        p.write_text(
            "\n".join([
                "plain line prompt",
                json.dumps({"prompt": "from json"}),
                '{"not valid json',          # malformed -> kept verbatim as a raw line
                json.dumps({"other": "x"}),  # JSON without a prompt key -> skipped
                "",
                json.dumps({"prompt": ""}),  # empty prompt -> skipped
            ]),
            encoding="utf-8",
        )
        assert ev._load_prompts(str(p), 10) == ["plain line prompt", "from json", '{"not valid json']

    def test_missing_and_empty_files_are_input_errors(self, tmp_path):
        with pytest.raises(UserInputError, match="Prompts file not found") as exc:
            ev._load_prompts(str(tmp_path / "gone.txt"), 5)
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        empty = tmp_path / "empty.txt"
        empty.write_text("\n\n", encoding="utf-8")
        with pytest.raises(UserInputError, match="no usable prompts"):
            ev._load_prompts(str(empty), 5)
        only_objects = tmp_path / "objs.jsonl"
        only_objects.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
        with pytest.raises(UserInputError, match="no usable prompts"):
            ev._load_prompts(str(only_objects), 5)


# =============================================================================
# Scripted stub model at the loading seam
# =============================================================================


class StubModel:
    """Deterministic stand-in: constant loss, generates ids [5, 6] ("a cat")."""

    def __init__(self, loss: float = 1.5, *, device=None, fail_generate: bool = False):
        self.loss = loss
        self.device = device
        self.fail_generate = fail_generate
        self.generate_kwargs: list[dict] = []
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        return types.SimpleNamespace(loss=torch.tensor(self.loss))

    def generate(self, input_ids, **kwargs):
        if self.fail_generate:
            raise RuntimeError("CUDA out of memory")
        self.generate_kwargs.append(kwargs)
        return torch.cat([input_ids, torch.tensor([[5, 6]])], dim=1)


@pytest.fixture
def stubbed(monkeypatch):
    calls = []
    holder = types.SimpleNamespace(model=StubModel(), calls=calls)

    def fake_load(run):
        calls.append(run["run_id"])
        return holder.model, tiny_tokenizer()

    monkeypatch.setattr(ev, "_load_model_and_tokenizer", fake_load)
    return holder


@pytest.fixture
def stub_run(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    RunHistoryManager(str(out)).record_run_started(run_id="stub-run", model_name="stub/model")
    return str(out)


class TestHeldOutLoss:
    def test_mean_is_per_sequence_not_token_weighted(self):
        model = StubModel(loss=2.0)
        tok = tiny_tokenizer()
        loss = ev._compute_held_out_loss(model, tok, ["the cat", "a dog ran to the park"], max_length=32)
        assert loss == pytest.approx(2.0)
        assert model.eval_calls == 1

    def test_empty_texts_return_none(self):
        assert ev._compute_held_out_loss(StubModel(), tiny_tokenizer(), [], max_length=8) is None

    def test_non_finite_losses_are_skipped_and_logged(self, caplog):
        losses = iter([float("nan"), 4.0, float("inf")])

        class Mixed(StubModel):
            def __call__(self, **kw):
                return types.SimpleNamespace(loss=torch.tensor(next(losses)))

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            loss = ev._compute_held_out_loss(Mixed(), tiny_tokenizer(), ["the cat", "a dog", "the mat"], max_length=8)
        assert loss == pytest.approx(4.0)
        assert sum("non-finite held-out loss" in r.getMessage() for r in caplog.records) == 2

    def test_all_non_finite_returns_none(self):
        assert ev._compute_held_out_loss(
            StubModel(loss=float("nan")), tiny_tokenizer(), ["the cat"], max_length=8
        ) is None

    def test_unmovable_device_leaves_inputs_on_their_original_device(self):
        """A model whose ``device`` cannot be honoured still gets a consistent pair."""
        seen = []

        class Recording(StubModel):
            def __call__(self, input_ids=None, attention_mask=None, labels=None):
                seen.append((input_ids.device.type, attention_mask.device.type))
                return types.SimpleNamespace(loss=torch.tensor(1.0))

        loss = ev._compute_held_out_loss(
            Recording(device="no-such-device"), tiny_tokenizer(), ["the cat sat"], max_length=8
        )
        assert loss == pytest.approx(1.0)
        assert seen == [("cpu", "cpu")]


class TestGenerate:
    def test_sampling_vs_greedy_kwargs_and_prompt_stripping(self):
        model = StubModel()
        tok = tiny_tokenizer()
        sampled = ev._generate(model, tok, ["the cat"], max_new_tokens=7, temperature=0.8, seed=1)
        greedy = ev._generate(model, tok, ["the cat"], max_new_tokens=7, temperature=0.0, seed=1)
        assert model.generate_kwargs == [
            {"max_new_tokens": 7, "do_sample": True, "temperature": 0.8},
            {"max_new_tokens": 7, "do_sample": False},
        ]
        assert sampled == greedy
        assert sampled[0].prompt == "the cat" and sampled[0].completion == "a cat"  # prompt prefix stripped

    def test_seed_failures_do_not_abort_generation(self, monkeypatch):
        def refuse(seed):
            raise RuntimeError("seeding unavailable")

        monkeypatch.setattr(torch, "manual_seed", refuse)
        out = ev._generate(StubModel(), tiny_tokenizer(), ["the cat"], max_new_tokens=2, temperature=0.5, seed=0)
        assert out[0].completion == "a cat"

    def test_device_move_failure_is_tolerated(self):
        out = ev._generate(
            StubModel(device="no-such-device"), tiny_tokenizer(), ["the cat"], max_new_tokens=2, temperature=0.0, seed=0
        )
        assert out[0].completion == "a cat"


class TestEvaluateRunWithStub:
    def test_known_completion_scores_exactly_against_references(self, stubbed, stub_run, tmp_path):
        refs = [
            {"prompt": "the cat", "reference": "a cat"},          # article dropped -> EM 1.0
            {"prompt": "a dog", "references": ["nope", "CAT!"]},  # punctuation/case -> EM 1.0
            {"prompt": "the mat", "reference": "cat sat down"},   # F1 = 2*(1*(1/3))/(1+1/3) = 0.5
        ]
        result = ev.evaluate_run(
            "stub-run", output_dir=stub_run, heldout_texts=["the cat sat"], n=1, references=refs
        )
        assert result.held_out_loss == pytest.approx(1.5)
        assert result.perplexity == pytest.approx(math.exp(1.5))
        assert result.eval_n == 3
        assert result.task_metrics["normalized_exact_match"] == pytest.approx(2 / 3)
        assert result.task_metrics["token_f1"] == pytest.approx((1 + 1 + 0.5) / 3)
        # reference prompts are generated greedily regardless of the caller's temperature
        greedy_calls = [k for k in stubbed.model.generate_kwargs if k["do_sample"] is False]
        assert len(greedy_calls) == 3

    def test_nan_loss_yields_no_perplexity(self, stubbed, stub_run):
        stubbed.model = StubModel(loss=float("nan"))
        result = ev.evaluate_run("stub-run", output_dir=stub_run, heldout_texts=["the cat"], n=1)
        assert result.held_out_loss is None and result.perplexity is None
        assert len(result.generations) == 1  # generations still populate

    def test_astronomical_loss_overflows_to_infinite_perplexity(self, stubbed, stub_run):
        stubbed.model = StubModel(loss=1000.0)
        result = ev.evaluate_run("stub-run", output_dir=stub_run, heldout_texts=["the cat"], n=1)
        assert result.held_out_loss == pytest.approx(1000.0)
        assert result.perplexity == math.inf

    def test_generation_crash_is_wrapped_with_the_stable_code(self, stubbed, stub_run):
        stubbed.model = StubModel(fail_generate=True)
        with pytest.raises(TrainingError) as exc:
            ev.evaluate_run("stub-run", output_dir=stub_run, heldout_texts=["the cat"], n=1)
        assert exc.value.code == "RUNTIME_EVAL_FAILED"
        assert "CUDA out of memory" in exc.value.message
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_reference_without_an_answer_is_an_input_error_not_a_runtime_failure(self, stubbed, stub_run):
        with pytest.raises(UserInputError) as exc:
            ev.evaluate_run(
                "stub-run", output_dir=stub_run, heldout_texts=["the cat"], n=1,
                references=[{"prompt": "q"}],
            )
        assert exc.value.code == "INPUT_VALIDATION_FAILED"
        assert "missing a 'reference'/'references' key" in exc.value.message

    def test_bad_inputs_fail_before_the_model_is_loaded(self, stubbed, stub_run):
        with pytest.raises(UserInputError, match="Unknown eval metric 'rouge'"):
            ev.evaluate_run(
                "stub-run", output_dir=stub_run, heldout_texts=["x y"], references=[{"prompt": "q", "reference": "a"}],
                metrics=["rouge"],
            )
        with pytest.raises(UserInputError, match="non-empty 'prompt'"):
            ev.evaluate_run(
                "stub-run", output_dir=stub_run, heldout_texts=["x y"], references=[{"reference": "a"}],
            )
        with pytest.raises(UserInputError, match="non-empty 'prompt'"):
            ev.evaluate_run(
                "stub-run", output_dir=stub_run, heldout_texts=["x y"], references=["not a dict"],  # type: ignore[list-item]
            )
        assert stubbed.calls == []  # the heavy load never happened

    def test_history_write_failure_never_sinks_the_eval(self, stubbed, stub_run, monkeypatch, caplog):
        def refuse(self, *a, **k):
            raise OSError("history file locked")

        monkeypatch.setattr(RunHistoryManager, "record_run_completed", refuse)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = ev.evaluate_run("stub-run", output_dir=stub_run, heldout_texts=["the cat"], n=1)
        assert result.held_out_loss == pytest.approx(1.5)
        assert any("failed to persist eval result" in r.getMessage() and "history file locked" in r.getMessage()
                   for r in caplog.records)


# =============================================================================
# Metric helpers: the branches not reached through evaluate_run
# =============================================================================


class TestMetricEdges:
    @pytest.fixture(autouse=True)
    def _allow_code_eval(self, monkeypatch):
        # pass_rate runs model output in a child process and is opt-in.
        monkeypatch.setenv("BACKPROPAGATE_ALLOW_CODE_EVAL", "1")

    def test_reference_coercion(self):
        assert ev._as_reference_list(None) == []
        assert ev._as_reference_list("one") == ["one"]
        assert ev._as_reference_list([1, "two"]) == ["1", "two"]

    @pytest.mark.parametrize("fn", [ev.normalized_exact_match, ev.token_f1, ev.contains_match, ev.regex_match, ev.pass_rate])
    def test_every_metric_scores_zero_without_references(self, fn):
        assert fn("anything", None) == 0.0
        assert fn("anything", []) == 0.0

    def test_squad_normalisation(self):
        assert ev.normalize_squad_text("The  Theme, of AN apple!") == "theme of apple"

    def test_f1_edge_cases(self):
        assert ev.token_f1("", [""]) == 1.0  # both empty = a correct "no answer"
        assert ev.token_f1("", ["x"]) == 0.0
        assert ev.token_f1("x", [""]) == 0.0
        assert ev.token_f1("cat sat", ["a cat sat down"]) == pytest.approx(0.8)
        assert ev.token_f1("cat cat", ["cat dog"]) == pytest.approx(0.5)  # multiset overlap
        assert ev.token_f1("apple", ["pear", "apple"]) == 1.0  # max over references

    def test_contains_and_regex(self):
        assert ev.contains_match("The ANSWER is 42.", ["answer is 42"]) == 1.0
        assert ev.contains_match("nope", ["yes", "no"]) == 1.0
        assert ev.regex_match("code 2026 ok", [r"\d{4}"]) == 1.0
        assert ev.regex_match("code ok", [r"\d{4}", "zzz"]) == 0.0
        with pytest.raises(UserInputError) as exc:
            ev.regex_match("x", ["(unclosed"])
        assert exc.value.code == "INPUT_VALIDATION_FAILED"

    def test_pass_rate(self):
        code = "def add(a, b):\n    return a + b\n"
        tests = ["assert add(1, 2) == 3", "assert add(2, 2) == 5", "assert add(0, 0) == 0"]
        assert ev.pass_rate(code, tests) == pytest.approx(2 / 3)
        assert ev.pass_rate("def broken(:", tests) == 0.0  # syntax error -> 0, no crash
        assert ev.pass_rate(code, "assert add(5, 5) == 10") == 1.0  # a bare string is one snippet
        assert ev.pass_rate("import os", ["assert True"]) == 0.0  # __import__ is not available

    def test_dispatch_and_unknown_metric(self):
        assert ev.compute_task_metric("contains", "abc", ["b"]) == 1.0
        with pytest.raises(UserInputError, match="Unknown eval metric 'bleu'") as exc:
            ev.compute_task_metric("bleu", "a", ["a"])
        assert "token_f1" in (exc.value.suggestion or exc.value.details.get("hint", "") or str(exc.value))

    def test_bootstrap_ci(self):
        assert ev.bootstrap_ci_halfwidth([]) is None
        assert ev.bootstrap_ci_halfwidth([1.0]) is None
        assert ev.bootstrap_ci_halfwidth([1.0, 1.0, 1.0]) == 0.0  # no variance -> zero-width
        scores = [1.0, 0.0] * 20
        a = ev.bootstrap_ci_halfwidth(scores, seed=5)
        b = ev.bootstrap_ci_halfwidth(scores, seed=5)
        assert a == b and 0.05 < a < 0.3  # a 50/50 mean over 40 items has a ~0.15 half-width
        assert ev.bootstrap_ci_halfwidth(scores, n_resamples=200, confidence=0.5, seed=5) < a

    def test_compute_task_metrics_shapes(self):
        gens = [ev.GenerationSample("p", "yes"), ev.GenerationSample("q", "no")]
        assert ev._compute_task_metrics(gens, [], ["contains"]) == ({}, 0, None)
        assert ev._compute_task_metrics(gens, [{"prompt": "p", "reference": "yes"}], []) == ({}, 0, None)
        with pytest.raises(UserInputError, match=r"#0 must be a dict"):
            ev._compute_task_metrics(gens, ["bad"], ["contains"])  # type: ignore[list-item]
        # more references than generations: only the aligned prefix is scored
        refs = [{"prompt": "p", "reference": "yes"}, {"prompt": "q", "reference": "x"}, {"prompt": "r", "reference": "z"}]
        metrics, n, ci = ev._compute_task_metrics(gens, refs, ["contains"])
        assert n == 2 and metrics == {"contains": 0.5} and set(ci) == {"contains"}
        # no generations at all: nothing to score, no metric keys emitted
        assert ev._compute_task_metrics([], refs, ["contains"]) == ({}, 0, None)
        # a single scored item has no CI
        _, n1, ci1 = ev._compute_task_metrics(gens[:1], refs[:1], ["contains"])
        assert n1 == 1 and ci1 is None


class TestDiffEvals:
    def test_rows_include_union_of_task_metrics_and_eval_n(self):
        a = ev.EvalResult("a", "m", 1.0, 2.7, task_metrics={"token_f1": 0.5}, eval_n=10)
        b = ev.EvalResult("b", "", None, None, task_metrics={"contains": 1.0}, eval_n=0)
        diff = ev.diff_evals(a, b)
        rows = {r[0]: (r[1], r[2]) for r in diff.rows}
        assert rows["model_name"] == ("m", "n/a")
        assert rows["held_out_loss"] == ("1.0000", "n/a")
        assert rows["contains"] == ("n/a", "1.0000") and rows["token_f1"] == ("0.5000", "n/a")
        assert rows["eval_n"] == ("10", "0")
        assert diff.to_dict()["rows"][0] == ["model_name", "m", "n/a"]
        no_n = ev.diff_evals(ev.EvalResult("a", "m", 1.0, 2.0), ev.EvalResult("b", "m", 1.0, 2.0))
        assert "eval_n" not in {r[0] for r in no_n.rows}
