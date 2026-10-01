"""Coverage tests for the module-level helpers of ``backpropagate.trainer``:
HF retry/classification, dataset hashing, chat-marker detection, the HF
callback bridge, full-FT ceilings and the FSDP runtime gate.

Mock boundary (named per test): the network (no HF calls are made; transient
errors are raised by the callable under test), CUDA / ``torch.distributed``
availability probes (hardware), and ``sys.modules`` entries set to ``None`` to
simulate a missing optional dependency. Real ``httpx.Response`` /
``HfHubHTTPError`` / ``requests`` exception objects and a real tiny tokenizer
are used wherever behaviour depends on them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import types
from pathlib import Path

import httpx
import pytest
import requests

from backpropagate import trainer as T
from backpropagate.exceptions import (
    FsdpUnavailableError,
    FullFinetuneModelTooLargeError,
)

LOGGER = "backpropagate.trainer"


def hf_error(status: int):
    """A real ``HfHubHTTPError`` carrying a real ``httpx.Response``."""
    from huggingface_hub.utils import HfHubHTTPError

    req = httpx.Request("GET", "https://huggingface.co/org/model")
    return HfHubHTTPError(f"HTTP {status}", response=httpx.Response(status, request=req))


def requests_http_error(status: int | None):
    resp = None
    if status is not None:
        resp = requests.Response()
        resp.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=resp)


# ---------------------------------------------------------------------------
# HF transient retry
# ---------------------------------------------------------------------------

class TestHfTransientExceptions:
    def test_full_candidate_set_when_deps_present(self):
        excs = T._hf_transient_exceptions()
        assert ConnectionError in excs and TimeoutError in excs
        assert requests.exceptions.HTTPError in excs
        from huggingface_hub.utils import HfHubHTTPError

        assert HfHubHTTPError in excs

    def test_degrades_without_requests_or_hub(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "requests", None)
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
        assert T._hf_transient_exceptions() == (ConnectionError, TimeoutError)


class TestIsTransient:
    def test_non_network_exception_is_not_transient(self):
        assert T._is_transient_hf_exception(ValueError("bad")) is False

    def test_connection_error_without_response_is_transient(self):
        assert T._is_transient_hf_exception(ConnectionError("reset")) is True
        assert T._is_transient_hf_exception(requests.ConnectionError("x")) is True

    @pytest.mark.parametrize("status,expected", [
        (429, True), (500, True), (503, True),
        (400, False), (401, False), (403, False), (404, False),
    ])
    def test_status_code_gates_retry(self, status, expected):
        assert T._is_transient_hf_exception(hf_error(status)) is expected
        assert T._is_transient_hf_exception(requests_http_error(status)) is expected

    def test_http_error_without_response_is_retried(self):
        assert T._is_transient_hf_exception(requests_http_error(None)) is True


class TestRetryHfCall:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        monkeypatch.setattr(T, "_RETRY_BASE_SECONDS", 0)
        monkeypatch.setattr(T, "_RETRY_MAX_SECONDS", 0)

    def test_returns_value_and_forwards_args(self):
        calls = []

        def fn(a, b=0):
            calls.append((a, b))
            return a + b

        assert T._retry_hf_call(fn, 1, b=2, _label="x") == 3
        assert calls == [(1, 2)]

    def test_retries_transient_then_succeeds_with_warning_logged(self, caplog):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise ConnectionError("503 blip")
            return "ok"

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert T._retry_hf_call(flaky, _label="model") == "ok"
        assert state["n"] == 3
        assert sum("Retrying" in r.getMessage() for r in caplog.records) == 2

    def test_exhaustion_logs_structured_line_and_reraises(self, caplog):
        state = {"n": 0}

        def always():
            state["n"] += 1
            raise TimeoutError("slow")

        with caplog.at_level(logging.ERROR, logger=LOGGER), pytest.raises(TimeoutError, match="slow"):
            T._retry_hf_call(always, _label="dataset_load")
        assert state["n"] == T._RETRY_ATTEMPTS
        assert "HF transient retry exhausted: label=dataset_load err=TimeoutError: slow" in caplog.text

    def test_non_transient_status_fails_fast_without_retry(self):
        state = {"n": 0}

        def gated():
            state["n"] += 1
            raise hf_error(403)

        from huggingface_hub.utils import HfHubHTTPError

        with pytest.raises(HfHubHTTPError):
            T._retry_hf_call(gated)
        assert state["n"] == 1

    def test_unrelated_exception_propagates_immediately(self):
        def bad():
            raise KeyError("k")

        with pytest.raises(KeyError):
            T._retry_hf_call(bad)


# ---------------------------------------------------------------------------
# ModelLoadError cause classification
# ---------------------------------------------------------------------------

class TestClassifyModelLoadCause:
    @pytest.mark.parametrize("status,cause", [
        (401, "auth"), (403, "auth"), (404, "not_found"), (500, "network"), (502, "network"),
    ])
    def test_hub_http_status_mapping(self, status, cause):
        assert T._classify_model_load_cause(hf_error(status)) == cause

    def test_hub_http_other_status_falls_through_to_unknown(self):
        assert T._classify_model_load_cause(hf_error(418)) == "unknown"

    @pytest.mark.parametrize("exc", [
        requests.ConnectionError("x"), requests.Timeout("t"), ConnectionError("c"),
        TimeoutError("t"),
    ])
    def test_network_family(self, exc):
        assert T._classify_model_load_cause(exc) == "network"

    @pytest.mark.parametrize("status,cause", [
        (401, "auth"), (403, "auth"), (404, "not_found"), (503, "network"),
    ])
    def test_requests_http_error_status_mapping(self, status, cause):
        assert T._classify_model_load_cause(requests_http_error(status)) == cause

    def test_requests_http_error_unknown_status_and_missing_response(self):
        assert T._classify_model_load_cause(requests_http_error(418)) == "unknown"
        assert T._classify_model_load_cause(requests_http_error(None)) == "unknown"

    def test_import_error_is_version(self):
        assert T._classify_model_load_cause(ImportError("cannot import name x")) == "version"

    def test_everything_else_is_unknown(self):
        assert T._classify_model_load_cause(RuntimeError("boom")) == "unknown"

    def test_without_optional_deps_still_classifies_builtins(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "requests", None)
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils", None)
        assert T._classify_model_load_cause(TimeoutError()) == "network"
        assert T._classify_model_load_cause(ImportError()) == "version"
        assert T._classify_model_load_cause(ValueError()) == "unknown"


# ---------------------------------------------------------------------------
# Dataset hash
# ---------------------------------------------------------------------------

class TestComputeDatasetHash:
    def test_non_path_inputs_return_none(self):
        assert T._compute_dataset_hash(None) is None
        assert T._compute_dataset_hash([{"a": 1}]) is None

    def test_missing_path_and_directory_return_none(self, tmp_path):
        assert T._compute_dataset_hash(str(tmp_path / "nope.jsonl")) is None
        assert T._compute_dataset_hash(tmp_path) is None

    def test_hashes_file_bytes_to_16_hex_chars(self, tmp_path):
        f = tmp_path / "d.jsonl"
        payload = b'{"a": 1}\n' * 200_000  # > 1 MiB so the chunk loop iterates
        f.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()[:16]
        assert T._compute_dataset_hash(f) == expected
        assert T._compute_dataset_hash(str(f)) == expected

    def test_unreadable_file_returns_none(self, tmp_path, monkeypatch, caplog):
        f = tmp_path / "d.jsonl"
        f.write_text("x")
        real_open = Path.open

        def deny(self, *a, **k):
            if self == f:
                raise PermissionError("locked")
            return real_open(self, *a, **k)

        monkeypatch.setattr(Path, "open", deny)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert T._compute_dataset_hash(f) is None
        assert "_compute_dataset_hash failed" in caplog.text


# ---------------------------------------------------------------------------
# Chat-marker detection
# ---------------------------------------------------------------------------

class _Tok:
    """Duck-typed tokenizer stand-in (the detector only reads attributes)."""

    def __init__(self, name_or_path=None, init_kwargs=None, render=None):
        if name_or_path is not None:
            self.name_or_path = name_or_path
        if init_kwargs is not None:
            self.init_kwargs = init_kwargs
        self._render = render

    def apply_chat_template(self, messages, tokenize=False):
        if isinstance(self._render, Exception):
            raise self._render
        return self._render


CHATML = ("<|im_start|>user", "<|im_start|>assistant")
LLAMA3 = ("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>")


class TestDetectChatMarkers:
    @pytest.mark.parametrize("name,expected", [
        ("meta-llama/Llama-3.2-1B-Instruct", LLAMA3),
        ("some/llama3-finetune", LLAMA3),
        ("google/gemma-2-9b", ("<start_of_turn>user", "<start_of_turn>model")),
        ("Qwen/Qwen2.5-7B", CHATML),
        ("microsoft/phi-3-mini", ("<|user|>", "<|assistant|>")),
        ("microsoft/Phi3-small", ("<|user|>", "<|assistant|>")),
        ("someone/my-chatml-model", CHATML),
    ])
    def test_family_table(self, name, expected):
        assert T._detect_chat_markers(_Tok(name_or_path=name)) == expected

    def test_mistral_matches_with_unsloth_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = T._detect_chat_markers(_Tok(name_or_path="mistralai/Mistral-7B-Instruct-v0.3"))
        assert out == ("[INST]", "[/INST]")
        assert "Mistral [INST]/[/INST] template may not mask" in caplog.text

    def test_name_taken_from_init_kwargs_dict(self):
        tok = _Tok(init_kwargs={"name_or_path": "Qwen/Qwen3-4B"})
        assert T._detect_chat_markers(tok) == CHATML

    def test_name_taken_from_init_kwargs_underscore_key(self):
        tok = _Tok(init_kwargs={"_name_or_path": "google/gemma-3-1b"})
        assert T._detect_chat_markers(tok) == ("<start_of_turn>user", "<start_of_turn>model")

    def test_empty_init_kwargs_name_continues_and_falls_to_default(self, caplog):
        tok = _Tok(init_kwargs={}, render=None)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert T._detect_chat_markers(tok) == CHATML
        assert "could not detect chat markers" in caplog.text

    def test_attribute_access_error_is_skipped(self):
        class Boom:
            @property
            def name_or_path(self):
                raise RuntimeError("lazy load failed")

            init_kwargs = {"name_or_path": "meta-llama/llama-3-8b"}

            def apply_chat_template(self, *a, **k):
                raise AssertionError("not reached")

        assert T._detect_chat_markers(Boom()) == LLAMA3

    def test_class_name_participates_in_matching(self):
        class QwenTokenizerFast:
            pass

        assert T._detect_chat_markers(QwenTokenizerFast()) == CHATML

    def test_probe_recognises_unbranded_chatml_template(self, caplog):
        rendered = (
            "<|im_start|>user\nZZZUSERPROBE<|im_end|>\n"
            "<|im_start|>assistant\nZZZASSISTANTPROBE<|im_end|>\n"
        )
        with caplog.at_level(logging.INFO, logger=LOGGER):
            out = T._detect_chat_markers(_Tok(name_or_path="acme/unbranded", render=rendered))
        assert out == CHATML
        assert "probe matched ChatML-shaped template" in caplog.text

    def test_probe_with_real_tokenizer_template(self):
        from tests.helpers.tiny_models import tiny_tokenizer

        tok = tiny_tokenizer()
        tok.chat_template = (
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        )
        assert T._detect_chat_markers(tok) == CHATML

    def test_probe_non_chatml_render_falls_back_with_warning(self, caplog):
        tok = _Tok(name_or_path="acme/other", render="[USER] ZZZUSERPROBE [BOT] ZZZASSISTANTPROBE")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert T._detect_chat_markers(tok) == CHATML
        assert "could not detect chat markers" in caplog.text

    def test_probe_failure_is_logged_and_falls_back(self, caplog):
        tok = _Tok(name_or_path="acme/other", render=ValueError("no template"))
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert T._detect_chat_markers(tok) == CHATML
        assert "apply_chat_template probe failed" in caplog.text

    def test_probe_returning_non_string_falls_back(self):
        assert T._detect_chat_markers(_Tok(name_or_path="acme/x", render=["tok", "ids"])) == CHATML


# ---------------------------------------------------------------------------
# HF callback bridge
# ---------------------------------------------------------------------------

class TestTrlBridgeCallback:
    def test_returns_none_when_transformers_lacks_trainer_callback(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert T._build_trl_bridge_callback(T.TrainingCallback()) is None
        assert "TrainerCallback unavailable" in caplog.text

    def _state(self, **kw):
        return types.SimpleNamespace(**kw)

    def test_adapter_is_a_real_hf_trainer_callback(self):
        from transformers import TrainerCallback

        assert isinstance(T._build_trl_bridge_callback(T.TrainingCallback()), TrainerCallback)

    def test_on_log_forwards_loss_and_step(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_step=lambda s, v: seen.append((s, v))))
        cb.on_log(None, self._state(global_step=7), None, logs={"loss": "0.5"})
        assert seen == [(7, 0.5)]

    def test_on_log_without_on_step_is_noop(self):
        cb = T._build_trl_bridge_callback(T.TrainingCallback())
        cb.on_log(None, self._state(global_step=1), None, logs={"loss": 1.0})  # no raise

    def test_on_log_falls_back_to_history_tail(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_step=lambda s, v: seen.append((s, v))))
        state = self._state(global_step=3, log_history=[
            {"loss": 0.9}, {"eval_loss": 0.1}, "junk", {"loss": "bad"}, {"loss": 0.4}, {"eval_loss": 0.2},
        ])
        cb.on_log(None, state, None, logs={"loss": "not-a-number"})
        assert seen == [(3, 0.4)]

    def test_on_log_history_entry_unparseable_then_older_valid(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_step=lambda s, v: seen.append(v)))
        state = self._state(global_step=1, log_history=[{"loss": 0.7}, {"loss": None}])
        cb.on_log(None, state, None, logs={"eval_loss": 1.0})
        assert seen == [0.7]

    def test_on_log_without_any_loss_is_skipped(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_step=lambda s, v: seen.append(v)))
        cb.on_log(None, self._state(global_step=1, log_history=[{"eval_loss": 1.0}]), None,
                  logs={"eval_loss": 1.0})
        cb.on_log(None, self._state(global_step=1), None, logs=None)
        assert seen == []

    def test_on_log_step_defaults_to_zero(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_step=lambda s, v: seen.append((s, v))))
        cb.on_log(None, self._state(), None, logs={"loss": 2.0})
        assert seen == [(0, 2.0)]

    def test_on_log_user_callback_error_is_isolated(self, caplog):
        def bad(step, loss):
            raise RuntimeError("ui died")

        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_step=bad))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            cb.on_log(None, self._state(global_step=4), None, logs={"loss": 1.25})
        assert "on_step callback raised error (step=4 loss=1.2500): ui died" in caplog.text

    def test_on_epoch_end_forwards_int_epoch(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_epoch=seen.append))
        cb.on_epoch_end(None, self._state(epoch=2.0), None)
        cb.on_epoch_end(None, self._state(epoch=None), None)
        cb.on_epoch_end(None, self._state(epoch="oops"), None)
        assert seen == [2, 0, 0]

    def test_on_epoch_end_noop_without_hook_and_isolates_errors(self, caplog):
        T._build_trl_bridge_callback(T.TrainingCallback()).on_epoch_end(
            None, self._state(epoch=1), None)

        def bad(epoch):
            raise ValueError("nope")

        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_epoch=bad))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            cb.on_epoch_end(None, self._state(epoch=3), None)
        assert "on_epoch callback raised error (epoch=3): nope" in caplog.text

    def test_on_save_builds_checkpoint_path(self, tmp_path):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_save=seen.append))
        cb.on_save(types.SimpleNamespace(output_dir=str(tmp_path)), self._state(global_step=50), None)
        assert seen == [str(tmp_path / "checkpoint-50")]

    def test_on_save_without_output_dir_or_with_bad_state_passes_empty_path(self):
        seen = []
        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_save=seen.append))
        cb.on_save(types.SimpleNamespace(output_dir=None), self._state(global_step=1), None)
        cb.on_save(types.SimpleNamespace(output_dir="/x"), self._state(global_step="abc"), None)
        assert seen == ["", ""]

    def test_on_save_noop_without_hook_and_isolates_errors(self, caplog):
        T._build_trl_bridge_callback(T.TrainingCallback()).on_save(
            types.SimpleNamespace(output_dir="/x"), self._state(global_step=1), None)

        def bad(path):
            raise OSError("remote copy failed")

        cb = T._build_trl_bridge_callback(T.TrainingCallback(on_save=bad))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            cb.on_save(types.SimpleNamespace(output_dir=None), self._state(global_step=1), None)
        assert "on_save callback raised error (path=''): remote copy failed" in caplog.text


# ---------------------------------------------------------------------------
# Full-FT ceilings
# ---------------------------------------------------------------------------

class TestCeilings:
    @pytest.mark.parametrize("vram,expected", [
        (None, 4.0), (8.0, 4.0), (16.0, 4.0), (22.6, 5.0), (24.0, 5.0),
        (31.8, 6.0), (32.0, 6.0), (47.0, 10.0), (80.0, 10.0),
    ])
    def test_pure_gpu_ceiling_tiers(self, vram, expected):
        assert T._full_ft_ceiling_for_vram(vram) == expected

    def test_negative_vram_hits_defensive_floor(self):
        assert T._full_ft_ceiling_for_vram(-10.0) == T._FULL_FT_PARAM_CEILING_BILLIONS

    @pytest.mark.parametrize("vram,embed,expected", [
        (None, True, 4.0), (16.0, True, 4.0), (24.0, True, 6.0), (32.0, True, 8.0), (48.0, True, 12.0),
        (16.0, False, 5.0), (24.0, False, 8.0), (32.0, False, 11.0), (48.0, False, 16.0),
    ])
    def test_block_engine_ceiling_tiers(self, vram, embed, expected):
        assert T._full_ft_block_ceiling_for_vram(vram, train_embeddings=embed) == expected

    def test_block_ceiling_negative_vram_floor(self):
        assert T._full_ft_block_ceiling_for_vram(-5.0) == T._FULL_FT_PARAM_CEILING_BILLIONS

    def test_offload_ceiling_uses_host_ram_probe(self, monkeypatch):
        from backpropagate import offload_engine as oe

        monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: (64.0, None))
        assert T._full_ft_offload_ceiling_billions() == T._FULL_FT_PARAM_CEILING_BILLIONS
        monkeypatch.setattr(oe, "detect_host_ram_gib", lambda: (64.0, 48.0))
        assert T._full_ft_offload_ceiling_billions() == oe.offload_param_ceiling_billions(48.0)


class TestDetectTotalVram:
    def test_no_cuda_returns_none(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert T._detect_total_vram_gb() is None

    def test_reports_gib(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties",
                            lambda i: types.SimpleNamespace(total_memory=24 * 1024**3))
        assert T._detect_total_vram_gb() == pytest.approx(24.0)

    def test_probe_error_returns_none(self, monkeypatch, caplog):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom(i):
            raise RuntimeError("driver hiccup")

        monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert T._detect_total_vram_gb() is None
        assert "VRAM probe failed" in caplog.text

    def test_torch_missing_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert T._detect_total_vram_gb() is None


# ---------------------------------------------------------------------------
# Param-count estimation + full-FT gate
# ---------------------------------------------------------------------------

class TestEstimateParamCount:
    @pytest.mark.parametrize("model_id,expected", [
        ("", None),
        ("Qwen/Qwen2.5-7B-Instruct", 7.0),          # preset by HF id
        ("phi-4-mini-3.8b", 3.8),                    # preset by name
        ("QWEN2.5-3B", 3.0),                         # preset name, case-insensitive
        ("acme/Custom-1.5B-chat", 1.5),              # regex fallback
        ("acme/model-0.5b", 0.5),
        ("acme/Model-13B-v2-4bit", 13.0),
        ("acme/no-size-here", None),
        ("acme/Huge-5000B", None),                   # outside sanity bound
        ("acme/Tiny-0.01B", None),
    ])
    def test_resolution(self, model_id, expected):
        assert T._estimate_param_count_billions(model_id) == expected

    def test_picks_largest_regex_match(self):
        assert T._estimate_param_count_billions("acme/phi-4b-mini-3.8b") == 4.0

    def test_preset_without_numeric_name_falls_through_to_regex(self, monkeypatch):
        from backpropagate import config

        preset = types.SimpleNamespace(name="weird-preset", model_id="acme/Weird-9B")
        monkeypatch.setattr(config, "lookup_model_preset_by_id", lambda mid: preset)
        assert T._estimate_param_count_billions("acme/Weird-9B") == 9.0

    def test_preset_size_outside_sanity_bound_is_ignored(self, monkeypatch):
        from backpropagate import config

        preset = types.SimpleNamespace(name="foo-5000b", model_id="acme/foo")
        monkeypatch.setattr(config, "lookup_model_preset_by_id", lambda mid: preset)
        assert T._estimate_param_count_billions("acme/foo") is None

    def test_preset_lookup_error_is_swallowed(self, monkeypatch, caplog):
        from backpropagate import config

        def boom(mid):
            raise RuntimeError("catalog corrupt")

        monkeypatch.setattr(config, "lookup_model_preset_by_id", boom)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert T._estimate_param_count_billions("acme/Model-2B") == 2.0
        assert "preset path raised" in caplog.text


class _Param:
    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


class _NoNumParams:
    """Model object with only ``parameters()`` (no ``num_parameters``)."""

    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class TestEnforceFullFtCeiling:
    def test_loaded_model_num_parameters_over_ceiling_raises(self):
        model = types.SimpleNamespace(num_parameters=lambda: 7_000_000_000)
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            T._enforce_full_ft_param_ceiling("acme/m", loaded_model=model)
        assert ei.value.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE"
        assert ei.value.param_count_billions == pytest.approx(7.0)
        assert ei.value.ceiling_billions == 4.0
        assert ei.value.offload_recoverable is False

    def test_manual_parameter_sum_when_no_num_parameters(self):
        model = _NoNumParams([_Param(3_000_000_000), _Param(2_500_000_000),
                              object()])  # object() has no numel -> counted as 0
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            T._enforce_full_ft_param_ceiling("acme/m", loaded_model=model)
        assert ei.value.param_count_billions == pytest.approx(5.5)

    def test_bad_param_is_skipped_not_fatal(self):
        class Bad:
            def numel(self):
                raise RuntimeError("meta tensor")

        model = _NoNumParams([Bad(), _Param(1_000_000_000)])
        T._enforce_full_ft_param_ceiling("acme/m", loaded_model=model)  # 1B <= 4B, no raise

    def test_zero_params_falls_back_to_model_id_heuristic(self):
        model = _NoNumParams([])
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            T._enforce_full_ft_param_ceiling("acme/Big-13B", loaded_model=model)
        assert ei.value.param_count_billions == 13.0

    def test_loaded_model_probe_error_falls_back_to_id(self, caplog):
        class Broken:
            def num_parameters(self):
                raise RuntimeError("no meta")

        with caplog.at_level(logging.DEBUG, logger=LOGGER), \
                pytest.raises(FullFinetuneModelTooLargeError):
            T._enforce_full_ft_param_ceiling("acme/Big-13B", loaded_model=Broken())
        assert "loaded model probe raised" in caplog.text

    def test_unknown_size_defers_with_info_log(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            T._enforce_full_ft_param_ceiling("acme/mystery")  # returns None, no raise
        assert "deferring the mode='full' check to load_model() time" in caplog.text

    def test_within_ceiling_approved_and_offload_suffix_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            T._enforce_full_ft_param_ceiling("acme/Small-1B", full_ft_offload=True)
        assert "mode='full' approved" in caplog.text
        assert "(FSDP2 CPU-offload)" in caplog.text

    def test_offload_recoverable_when_between_ceilings_and_offload_off(self):
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            T._enforce_full_ft_param_ceiling(
                "acme/Mid-7B", offload_ceiling_billions=8.0, full_ft_offload=False)
        assert ei.value.offload_recoverable is True
        assert ei.value.offload_active is False
        assert "full_ft_offload" in str(ei.value) or "full-ft-offload" in str(ei.value)

    def test_not_recoverable_when_offload_already_on_or_beyond_offload_ceiling(self):
        with pytest.raises(FullFinetuneModelTooLargeError) as ei:
            T._enforce_full_ft_param_ceiling(
                "acme/Mid-7B", offload_ceiling_billions=8.0, full_ft_offload=True)
        assert ei.value.offload_recoverable is False and ei.value.offload_active is True
        with pytest.raises(FullFinetuneModelTooLargeError) as ei2:
            T._enforce_full_ft_param_ceiling(
                "acme/Huge-70B", offload_ceiling_billions=8.0, full_ft_offload=False)
        assert ei2.value.offload_recoverable is False


# ---------------------------------------------------------------------------
# FSDP runtime
# ---------------------------------------------------------------------------

@pytest.fixture
def fsdp_env(monkeypatch):
    """Isolate the env vars ``_ensure_fsdp_runtime`` seeds via ``setdefault``."""
    for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)


class TestEnsureFsdpRuntime:
    def _patch(self, monkeypatch, *, cuda=True, dist_ok=True, nccl=True, initialized=False,
               init_exc=None):
        import torch
        import torch.distributed as dist

        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(dist, "is_available", lambda: dist_ok)
        monkeypatch.setattr(dist, "is_nccl_available", lambda: nccl)
        monkeypatch.setattr(dist, "is_initialized", lambda: initialized)

        def init(**kw):
            calls.append(kw)
            if init_exc is not None:
                raise init_exc

        monkeypatch.setattr(dist, "init_process_group", init)
        return calls

    def test_torch_distributed_import_failure(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch.distributed", None)
        with pytest.raises(FsdpUnavailableError) as ei:
            T._ensure_fsdp_runtime()
        assert ei.value.code == "DEP_FSDP_UNAVAILABLE"
        assert "torch.distributed is not importable" in ei.value.reason

    def test_no_cuda(self, monkeypatch, fsdp_env):
        self._patch(monkeypatch, cuda=False)
        with pytest.raises(FsdpUnavailableError, match="no CUDA device"):
            T._ensure_fsdp_runtime()

    @pytest.mark.parametrize("dist_ok,nccl", [(False, True), (True, False)])
    def test_nccl_unavailable_names_wsl2(self, monkeypatch, fsdp_env, dist_ok, nccl):
        self._patch(monkeypatch, dist_ok=dist_ok, nccl=nccl)
        with pytest.raises(FsdpUnavailableError) as ei:
            T._ensure_fsdp_runtime()
        assert "NCCL is unavailable" in ei.value.reason and "WSL2" in ei.value.reason

    def test_process_group_init_failure_is_structured(self, monkeypatch, fsdp_env):
        self._patch(monkeypatch, init_exc=RuntimeError("address in use"))
        with pytest.raises(FsdpUnavailableError, match="could not initialize the FSDP process group") as ei:
            T._ensure_fsdp_runtime()
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_success_initialises_single_process_group_and_warns(self, monkeypatch, fsdp_env, caplog):
        calls = self._patch(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            T._ensure_fsdp_runtime()
        assert calls == [{"backend": "nccl", "rank": 0, "world_size": 1}]
        assert os.environ["MASTER_ADDR"] == "127.0.0.1"
        assert os.environ["WORLD_SIZE"] == "1"
        assert "FSDP2 CPU-offload active" in caplog.text

    def test_already_initialised_group_is_left_alone(self, monkeypatch, fsdp_env, caplog):
        calls = self._patch(monkeypatch, initialized=True)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            T._ensure_fsdp_runtime()
        assert calls == []
        assert "MASTER_ADDR" not in os.environ
        assert "FSDP2 CPU-offload active" in caplog.text


class TestGatherFsdpStateDict:
    def test_returns_none_when_fsdp2_not_importable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", None)
        assert T._gather_fsdp_full_state_dict(object()) is None

    def test_returns_none_for_plain_module(self):
        import torch

        assert T._gather_fsdp_full_state_dict(torch.nn.Linear(2, 2)) is None

    def test_gathers_full_cpu_state_dict_for_fsdp_module(self, monkeypatch):
        import torch
        from torch.distributed.checkpoint import state_dict as ckpt_sd
        from torch.distributed.fsdp import FSDPModule

        # FSDP2's fully_shard swaps the module's class in place; mimic that.
        Sharded = type("FSDPLinear", (FSDPModule, torch.nn.Linear), {})
        model = torch.nn.Linear(2, 2)
        model.__class__ = Sharded
        seen = {}

        def fake_get(module, options=None):
            seen["module"] = module
            seen["options"] = options
            return {"weight": "gathered"}

        monkeypatch.setattr(ckpt_sd, "get_model_state_dict", fake_get)
        out = T._gather_fsdp_full_state_dict(model)
        assert out == {"weight": "gathered"}
        assert seen["module"] is model
        assert seen["options"].full_state_dict is True and seen["options"].cpu_offload is True
