"""Unit tests for the MLX / Apple-Silicon training backend (v1.5 T3.1).

These tests are CPU / any-OS and run with ``mlx_lm`` + the host platform
MOCKED — they NEVER require Apple hardware or an mlx install. The real-hardware
end-to-end smoke (which SKIPS on non-Apple hosts) is owned by a sibling agent;
THIS file pins the pure / mockable contracts:

* backend selection truth table (resolve_backend + the Trainer construction
  guards),
* the LOAD-BEARING PEFT-alpha → mlx-absolute-``scale`` mapping in
  ``MLXBackend.build_config``,
* ``build_argv`` shape,
* ``prepare_mlx_data_dir`` real-tmp-path IO (one JSON object per line, each
  round-tripping to ``{"messages": [...]}``; valid.jsonl split),
* the MLX unsupported-feature gates (orpo / fp8 / mode='full' raise;
  multi_run raises),
* the subprocess seam (mocked CompletedProcess → MLXRunResult; CalledProcessError
  → structured TrainingError; missing mlx → DEP_MLX_UNAVAILABLE),
* the BUILT-BUT-UNVERIFIED invariant that importing the module does NOT import
  ``mlx_lm`` (subprocess-only).
"""

import json
import subprocess
import sys
from unittest import mock

import pytest

from backpropagate.exceptions import (
    InvalidSettingError,
    MLXUnavailableError,
    TrainingError,
)
from backpropagate.mlx_backend import (
    MLXBackend,
    MLXRunResult,
    detect_apple_silicon,
    prepare_mlx_data_dir,
    resolve_backend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(**overrides):
    """Construct an MLXBackend with sane defaults for config/argv tests."""
    kwargs = {
        "model": "mlx-community/Qwen2.5-7B-Instruct",
        "dataset_dir": "/tmp/mlx_data",  # noqa: S108 — test path, never written here
        "adapter_path": "/tmp/mlx_adapter",  # noqa: S108
        "lora_r": 256,
        "lora_alpha": 512,
        "lora_dropout": 0.05,
        "learning_rate": 1e-5,
        "iters": 100,
        "batch_size": 2,
        "max_seq_length": 2048,
    }
    kwargs.update(overrides)
    return MLXBackend(**kwargs)


def _apple_mlx_env():
    """Context manager stack: mock the host as Apple Silicon WITH mlx present.

    Patches ``platform.system`` / ``platform.machine`` (consulted inside
    ``detect_apple_silicon``) and ``backpropagate.mlx_backend.check_feature``
    (the mlx feature flag) so the MLX rail is selected without real hardware.
    """
    return [
        mock.patch("platform.system", return_value="Darwin"),
        mock.patch("platform.machine", return_value="arm64"),
        mock.patch("backpropagate.mlx_backend.check_feature", return_value=True),
    ]


class _Apple:
    """Reusable Apple+mlx mock context (enter/exit the patch stack)."""

    def __enter__(self):
        self._patches = _apple_mlx_env()
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


# ---------------------------------------------------------------------------
# detect_apple_silicon + resolve_backend truth table
# ---------------------------------------------------------------------------

class TestBackendDetection:
    def test_detect_true_on_apple_with_mlx(self):
        with _Apple():
            assert detect_apple_silicon() is True

    def test_detect_false_on_apple_without_mlx(self):
        with mock.patch("platform.system", return_value="Darwin"), \
             mock.patch("platform.machine", return_value="arm64"), \
             mock.patch("backpropagate.mlx_backend.check_feature", return_value=False):
            assert detect_apple_silicon() is False

    def test_detect_false_on_intel_mac(self):
        with mock.patch("platform.system", return_value="Darwin"), \
             mock.patch("platform.machine", return_value="x86_64"), \
             mock.patch("backpropagate.mlx_backend.check_feature", return_value=True):
            assert detect_apple_silicon() is False

    def test_detect_false_on_windows(self):
        with mock.patch("platform.system", return_value="Windows"), \
             mock.patch("platform.machine", return_value="AMD64"), \
             mock.patch("backpropagate.mlx_backend.check_feature", return_value=True):
            assert detect_apple_silicon() is False

    def test_resolve_auto_apple_mlx(self):
        with _Apple():
            assert resolve_backend("auto") == "mlx"

    def test_resolve_auto_non_apple_cuda(self):
        # Real host here is non-Apple; auto must resolve to cuda.
        with mock.patch("backpropagate.mlx_backend.detect_apple_silicon", return_value=False):
            assert resolve_backend("auto") == "cuda"

    def test_resolve_auto_apple_no_mlx_cuda(self):
        with mock.patch("platform.system", return_value="Darwin"), \
             mock.patch("platform.machine", return_value="arm64"), \
             mock.patch("backpropagate.mlx_backend.check_feature", return_value=False):
            assert resolve_backend("auto") == "cuda"

    def test_resolve_cuda_passthrough(self):
        assert resolve_backend("cuda") == "cuda"

    def test_resolve_mlx_passthrough(self):
        # Validity of a forced "mlx" is enforced at the call site, NOT here.
        assert resolve_backend("mlx") == "mlx"


# ---------------------------------------------------------------------------
# Trainer-level backend selection + guards
# ---------------------------------------------------------------------------

class TestTrainerBackendSelection:
    def test_auto_non_apple_resolves_cuda(self):
        from backpropagate.trainer import Trainer

        with mock.patch("torch.cuda.is_available", return_value=False):
            t = Trainer(backend="auto", model="x")
        assert t._effective_backend == "cuda"

    def test_explicit_cuda_resolves_cuda(self):
        from backpropagate.trainer import Trainer

        with mock.patch("torch.cuda.is_available", return_value=False):
            t = Trainer(backend="cuda", model="x")
        assert t._effective_backend == "cuda"

    def test_auto_apple_resolves_mlx(self):
        from backpropagate.trainer import Trainer

        with _Apple():
            t = Trainer(backend="auto", model="mlx-community/x")
        assert t._effective_backend == "mlx"

    def test_forced_mlx_non_apple_raises(self):
        from backpropagate.trainer import Trainer

        with mock.patch("backpropagate.trainer.detect_apple_silicon", return_value=False):
            with pytest.raises(InvalidSettingError) as ei:
                Trainer(backend="mlx", model="x")
        assert ei.value.code == "CONFIG_INVALID_SETTING"

    def test_invalid_backend_value_raises(self):
        from backpropagate.trainer import Trainer

        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(InvalidSettingError) as ei:
                Trainer(backend="zzz", model="x")
        assert ei.value.code == "CONFIG_INVALID_SETTING"

    def test_boundary_train_routes_to_mlx_without_load_model(self):
        """The load-bearing boundary: Apple+mlx mocked, _train_with_mlx patched.

        ``Trainer(backend="auto").train(ds)`` must return a TrainingRun, must
        NOT raise GPUNotAvailableError, and must NOT call load_model() (the MLX
        rail never loads a CUDA model into this process).
        """
        from backpropagate.trainer import Trainer, TrainingRun

        with _Apple():
            t = Trainer(backend="auto", model="mlx-community/x")
            assert t._effective_backend == "mlx"

            sentinel = TrainingRun(run_id="r", steps=1, final_loss=0.5)
            with mock.patch.object(
                Trainer, "_train_with_mlx", return_value=sentinel
            ) as m_train, mock.patch.object(
                Trainer, "load_model",
                side_effect=AssertionError("load_model must NOT be called on MLX"),
            ) as m_load:
                out = t.train("data.jsonl", steps=5)

        assert isinstance(out, TrainingRun)
        assert out.run_id == "r"
        assert m_train.called
        assert not m_load.called


# ---------------------------------------------------------------------------
# MLX unsupported-feature gates
# ---------------------------------------------------------------------------

class TestMLXUnsupportedGates:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"backend": "mlx", "method": "orpo", "model": "m"},
            {"backend": "mlx", "fp8": True, "model": "m"},
            {"backend": "mlx", "mode": "full", "model": "m"},
        ],
        ids=["mlx+orpo", "mlx+fp8", "mlx+full"],
    )
    def test_unsupported_combinations_raise(self, kwargs):
        from backpropagate.trainer import Trainer

        with _Apple():
            with pytest.raises(InvalidSettingError) as ei:
                Trainer(**kwargs)
        assert ei.value.code == "CONFIG_INVALID_SETTING"

    def test_rslora_warns_and_ignores(self):
        """use_rslora on MLX must WARN-and-ignore (construct fine, not raise)."""
        from backpropagate.trainer import Trainer

        with _Apple():
            t = Trainer(backend="mlx", use_rslora=True, model="m")
        assert t._effective_backend == "mlx"

    def test_multi_run_on_mlx_raises(self):
        from backpropagate.trainer import Trainer

        with _Apple():
            t = Trainer(backend="mlx", model="m")
            with pytest.raises(InvalidSettingError) as ei:
                t.multi_run("data.jsonl")
        assert ei.value.code == "CONFIG_INVALID_SETTING"


# ---------------------------------------------------------------------------
# build_config — the LOAD-BEARING scale mapping
# ---------------------------------------------------------------------------

class TestBuildConfig:
    @pytest.mark.parametrize(
        "r,alpha,expected_scale",
        [
            (256, 512, 2.0),
            (16, 32, 2.0),
            (8, 32, 4.0),
            (64, 16, 0.25),
            (128, 128, 1.0),
        ],
    )
    def test_scale_is_alpha_over_r(self, r, alpha, expected_scale):
        cfg = _make_backend(lora_r=r, lora_alpha=alpha).build_config()
        lp = cfg["lora_parameters"]
        assert lp["rank"] == r
        assert lp["scale"] == pytest.approx(expected_scale)

    def test_dropout_and_fields(self):
        cfg = _make_backend(lora_dropout=0.1).build_config()
        lp = cfg["lora_parameters"]
        assert lp["dropout"] == pytest.approx(0.1)
        assert cfg["fine_tune_type"] == "lora"
        assert cfg["train"] is True
        assert cfg["iters"] == 100
        assert cfg["batch_size"] == 2
        assert cfg["max_seq_length"] == 2048
        assert cfg["adapter_path"] == "/tmp/mlx_adapter"  # noqa: S108
        assert cfg["data"] == "/tmp/mlx_data"  # noqa: S108

    def test_keys_omitted_uses_mlx_default(self):
        """lora_parameters.keys must be OMITTED (mlx default target)."""
        cfg = _make_backend().build_config()
        assert "keys" not in cfg["lora_parameters"]


# ---------------------------------------------------------------------------
# build_argv shape
# ---------------------------------------------------------------------------

class TestBuildArgv:
    def test_argv_shape(self):
        argv = MLXBackend.build_argv("/tmp/cfg.yaml")  # noqa: S108
        assert argv == ["mlx_lm.lora", "--train", "--config", "/tmp/cfg.yaml"]

    def test_argv_accepts_path(self, tmp_path):
        cfg = tmp_path / "lora_config.yaml"
        argv = MLXBackend.build_argv(cfg)
        assert argv[0] == "mlx_lm.lora"
        assert argv[-1] == str(cfg)

    def test_write_config_roundtrips_to_yaml(self, tmp_path):
        b = _make_backend()
        path = b.write_config(tmp_path / "lora_config.yaml")
        assert path.exists()
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded["lora_parameters"]["rank"] == 256
        assert loaded["lora_parameters"]["scale"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# prepare_mlx_data_dir — real tmp_path IO
# ---------------------------------------------------------------------------

class TestPrepareDataDir:
    def _read_jsonl(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_sharegpt_one_object_per_line(self, tmp_path):
        samples = [
            {"conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello"},
            ]},
            {"conversations": [
                {"from": "human", "value": "bye"},
                {"from": "gpt", "value": "goodbye"},
            ]},
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d", seed=1)
        train = out / "train.jsonl"
        assert train.exists()
        assert not (out / "valid.jsonl").exists()
        recs = self._read_jsonl(train)
        assert len(recs) == 2
        for rec in recs:
            assert list(rec.keys()) == ["messages"]
            assert all("role" in m and "content" in m for m in rec["messages"])

    def test_alpaca_roundtrips_to_messages(self, tmp_path):
        samples = [
            {"instruction": "Add", "input": "2+2", "output": "4"},
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d")
        recs = self._read_jsonl(out / "train.jsonl")
        assert len(recs) == 1
        msgs = recs[0]["messages"]
        assert msgs[-1]["role"] == "assistant"
        assert "4" in msgs[-1]["content"]

    def test_chatml_roundtrips_to_messages(self, tmp_path):
        # OpenAI/ChatML message-list input.
        samples = [
            {"messages": [
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": "pong"},
            ]},
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d")
        recs = self._read_jsonl(out / "train.jsonl")
        assert recs[0]["messages"][0]["role"] == "user"
        assert recs[0]["messages"][0]["content"] == "ping"
        assert recs[0]["messages"][1]["content"] == "pong"

    def test_valid_split_written(self, tmp_path):
        samples = [
            {"messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]}
            for i in range(10)
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d", valid_fraction=0.2, seed=7)
        train = self._read_jsonl(out / "train.jsonl")
        valid = self._read_jsonl(out / "valid.jsonl")
        assert len(valid) == 2
        assert len(train) == 8

    def test_each_line_is_single_json_object(self, tmp_path):
        """No pretty-printing / no whole-list dump: each line parses standalone."""
        samples = [
            {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
            {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d")
        raw = (out / "train.jsonl").read_text(encoding="utf-8")
        lines = raw.splitlines()
        assert len(lines) == 2
        # Each line must be a dict (object), not a fragment of a multi-line array.
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_max_samples_caps(self, tmp_path):
        samples = [
            {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
            for i in range(20)
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d", max_samples=5, seed=3)
        recs = self._read_jsonl(out / "train.jsonl")
        assert len(recs) == 5

    # -- re-audit #10: trace-length filtering on the MLX rail -----------------

    def test_reasoning_trace_filter_drops_non_think_rows(self, tmp_path):
        """reasoning_trace=True runs the <think> trace-length filter on this
        rail: a row WITH a sufficiently-long <think> span is kept; a no-think
        row (require_think) is dropped — making the trace knobs LIVE on MLX."""
        samples = [
            {"messages": [
                {"role": "user", "content": "solve 2+2"},
                # ~12-word think span; approx counter (~4 chars/token) puts it
                # comfortably above the default min_trace_tokens floor of 8.
                {"role": "assistant", "content":
                    "<think>add the two numbers carefully one step at a "
                    "time to be sure</think>4"},
            ]},
            {"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},  # no <think> → dropped
            ]},
        ]
        out = prepare_mlx_data_dir(
            samples, tmp_path / "d", reasoning_trace=True, shuffle=False
        )
        recs = self._read_jsonl(out / "train.jsonl")
        # Only the reasoning row survives.
        assert len(recs) == 1
        joined = "".join(m["content"] for m in recs[0]["messages"])
        assert "<think>" in joined

    def test_reasoning_trace_off_keeps_all_rows(self, tmp_path):
        """Without reasoning_trace the filter is a no-op (byte-identical to the
        pre-#10 behavior): a no-think row is NOT dropped."""
        samples = [
            {"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]},
        ]
        out = prepare_mlx_data_dir(samples, tmp_path / "d")  # reasoning_trace default False
        recs = self._read_jsonl(out / "train.jsonl")
        assert len(recs) == 1

    def test_reasoning_trace_max_bound_drops_overlong(self, tmp_path):
        """max_trace_tokens is a LIVE knob on this rail: an over-long trace is
        dropped, leaving 0 records → structured DatasetFormatError (no write)."""
        from backpropagate.exceptions import DatasetFormatError

        samples = [
            {"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content":
                    "<think>" + ("word " * 200) + "</think>answer"},
            ]},
        ]
        with pytest.raises(DatasetFormatError) as ei:
            prepare_mlx_data_dir(
                samples, tmp_path / "d", reasoning_trace=True, max_trace_tokens=8
            )
        assert ei.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"
        # Nothing was written.
        assert not (tmp_path / "d" / "train.jsonl").exists()

    def test_zero_records_raises_before_write(self, tmp_path):
        """re-audit LOW: a dataset that yields 0 parseable chat turns raises a
        structured DatasetFormatError BEFORE writing a 0-byte train.jsonl (which
        would surface later as an opaque mlx_lm.lora subprocess error)."""
        from backpropagate.exceptions import DatasetFormatError

        # Empty-string raw-text rows produce no ChatML turn → 0 records.
        samples = [{"text": ""}, {"text": ""}]
        with pytest.raises(DatasetFormatError) as ei:
            prepare_mlx_data_dir(samples, tmp_path / "d")
        assert ei.value.code == "INPUT_DATASET_FORMAT_UNSUPPORTED"
        assert "0 usable records" in str(ei.value)
        assert not (tmp_path / "d" / "train.jsonl").exists()


# ---------------------------------------------------------------------------
# The subprocess seam — run() / fuse()
# ---------------------------------------------------------------------------

class TestRunSeam:
    def test_run_parses_loss_from_stdout(self, tmp_path):
        b = _make_backend(adapter_path=tmp_path / "adapter", iters=10)
        fake_stdout = (
            "Iter 1: Train loss 2.500, Learning Rate 1.0e-05\n"
            "Iter 5: Train loss 1.800, Val loss 1.900\n"
            "Iter 10: Train loss 1.200\n"
        )
        completed = subprocess.CompletedProcess(
            args=["mlx_lm.lora"], returncode=0, stdout=fake_stdout, stderr=""
        )
        with mock.patch(
            "backpropagate.mlx_backend.check_feature", return_value=True
        ), mock.patch(
            "backpropagate.export._run_subprocess_interruptible",
            return_value=completed,
        ) as m_run:
            result = b.run()

        assert isinstance(result, MLXRunResult)
        assert result.final_loss == pytest.approx(1.2)  # LAST train loss
        assert result.val_loss == pytest.approx(1.9)
        assert result.iters == 10
        assert result.adapter_path == str(tmp_path / "adapter")
        # The seam was driven with the mlx_lm.lora argv.
        argv = m_run.call_args.args[0]
        assert argv[0] == "mlx_lm.lora"
        # And mlx_lm was NEVER imported.
        assert "mlx_lm" not in sys.modules

    def test_run_parse_miss_yields_none(self, tmp_path):
        b = _make_backend(adapter_path=tmp_path / "adapter")
        completed = subprocess.CompletedProcess(
            args=["mlx_lm.lora"], returncode=0,
            stdout="some output with no recognizable loss line\n", stderr="",
        )
        with mock.patch(
            "backpropagate.mlx_backend.check_feature", return_value=True
        ), mock.patch(
            "backpropagate.export._run_subprocess_interruptible",
            return_value=completed,
        ):
            result = b.run()
        assert result.final_loss is None  # parse miss → None, NOT a failure

    def test_run_called_process_error_wrapped_as_training_error(self, tmp_path):
        b = _make_backend(adapter_path=tmp_path / "adapter")
        err = subprocess.CalledProcessError(
            returncode=1, cmd=["mlx_lm.lora"], output="", stderr="boom: model not found"
        )
        with mock.patch(
            "backpropagate.mlx_backend.check_feature", return_value=True
        ), mock.patch(
            "backpropagate.export._run_subprocess_interruptible",
            side_effect=err,
        ):
            with pytest.raises(TrainingError) as ei:
                b.run()
        assert ei.value.code == "RUNTIME_TRAINING_FAILED"
        # The original CalledProcessError is chained.
        assert isinstance(ei.value.__cause__, subprocess.CalledProcessError)

    def test_run_raises_dep_mlx_unavailable_when_feature_absent(self, tmp_path):
        b = _make_backend(adapter_path=tmp_path / "adapter")
        with mock.patch(
            "backpropagate.mlx_backend.check_feature", return_value=False
        ):
            with pytest.raises(MLXUnavailableError) as ei:
                b.run()
        assert ei.value.code == "DEP_MLX_UNAVAILABLE"
        assert ei.value.retryable is False

    def test_fuse_builds_argv_and_runs(self, tmp_path):
        b = _make_backend(adapter_path=tmp_path / "adapter")
        completed = subprocess.CompletedProcess(
            args=["mlx_lm.fuse"], returncode=0, stdout="", stderr=""
        )
        with mock.patch(
            "backpropagate.mlx_backend.check_feature", return_value=True
        ), mock.patch(
            "backpropagate.export._run_subprocess_interruptible",
            return_value=completed,
        ) as m_run:
            out = b.fuse(tmp_path / "merged", export_gguf=True, gguf_path=tmp_path / "m.gguf")
        assert out == str(tmp_path / "merged")
        argv = m_run.call_args.args[0]
        assert argv[0] == "mlx_lm.fuse"
        assert "--export-gguf" in argv
        assert "--gguf-path" in argv


# ---------------------------------------------------------------------------
# re-audit LOW: _train_with_mlx must NOT coerce an unparsed loss to 0.0
# ---------------------------------------------------------------------------

class TestMLXTrainLossHonesty:
    def test_unparsed_loss_is_not_zero(self, tmp_path):
        """A successful MLX run whose stdout loss could not be parsed
        (``MLXRunResult.final_loss is None``) must NOT be recorded as 0.0 (which
        reads as a perfect run). The TrainingRun carries nan (the typed-float
        sentinel), and metadata['final_loss_parsed'] is False."""
        import math

        from backpropagate.trainer import Trainer, TrainingRun

        with _Apple():
            t = Trainer(backend="mlx", model="mlx-community/x", output_dir=str(tmp_path))
            unparsed = MLXRunResult(
                adapter_path=str(tmp_path / "mlx_adapter"),
                final_loss=None,  # the parse-miss case
                iters=5,
                raw_stdout="(no recognizable loss line)",
                val_loss=None,
            )
            with mock.patch(
                "backpropagate.mlx_backend.prepare_mlx_data_dir",
                return_value=tmp_path / "mlx_data",
            ), mock.patch.object(MLXBackend, "run", return_value=unparsed):
                run = t._train_with_mlx(
                    "data.jsonl", steps=5, samples=None, callback=None
                )

        assert isinstance(run, TrainingRun)
        # NOT 0.0 — nan is the honest "ran, loss unknown" sentinel.
        assert math.isnan(run.final_loss)
        assert run.metadata["final_loss_parsed"] is False

    def test_parsed_loss_is_preserved(self, tmp_path):
        """The happy path is unchanged: a parsed numeric loss flows verbatim."""
        from backpropagate.trainer import Trainer

        with _Apple():
            t = Trainer(backend="mlx", model="mlx-community/x", output_dir=str(tmp_path))
            parsed = MLXRunResult(
                adapter_path=str(tmp_path / "mlx_adapter"),
                final_loss=1.234,
                iters=5,
                raw_stdout="Iter 5: Train loss 1.234",
                val_loss=None,
            )
            with mock.patch(
                "backpropagate.mlx_backend.prepare_mlx_data_dir",
                return_value=tmp_path / "mlx_data",
            ), mock.patch.object(MLXBackend, "run", return_value=parsed):
                run = t._train_with_mlx(
                    "data.jsonl", steps=5, samples=None, callback=None
                )
        assert run.final_loss == pytest.approx(1.234)
        assert run.metadata["final_loss_parsed"] is True


# ---------------------------------------------------------------------------
# BUILT-BUT-UNVERIFIED invariant: no mlx_lm import on module load
# ---------------------------------------------------------------------------

class TestNoMlxImport:
    def test_importing_module_does_not_import_mlx_lm(self):
        """Importing backpropagate.mlx_backend must NOT import mlx_lm.

        The whole rail is subprocess-only so the module imports cleanly on a
        Windows / CUDA host where mlx is absent. We pop any cached entry, fresh-
        import the module, and assert mlx_lm did not get pulled in.
        """
        sys.modules.pop("backpropagate.mlx_backend", None)
        sys.modules.pop("mlx_lm", None)
        import importlib

        importlib.import_module("backpropagate.mlx_backend")
        assert "mlx_lm" not in sys.modules

    def test_module_source_has_no_mlx_lm_import_statement(self):
        """Static guard: no ACTUAL ``import mlx_lm`` / ``from mlx_lm`` statement.

        Scans real code lines (stripped, comments excluded) rather than the
        whole blob so the docstring's prose ("never does ``import mlx_lm``")
        does not trip the check. The toolchain must only ever be a subprocess.
        """
        import backpropagate.mlx_backend as m

        with open(m.__file__, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        offending = [
            line
            for line in lines
            if line.strip().startswith(("import mlx_lm", "from mlx_lm"))
        ]
        assert offending == [], f"found mlx_lm import statement(s): {offending}"
