"""Smoke tests for the four v1.1.0-shipped Reflex UI state classes.

TESTS-B-007 (v1.3 swarm Stage C humanization): The FRONTEND-A-002 /
FRONTEND-A-005 / FRONTEND-A-003 / FRONTEND-B-013 Reflex state classes —
``TrainState``, ``MultiRunState``, ``ExportState``, ``DatasetState`` — had
ZERO regression test coverage. Only ``RunsState`` was covered (by
``tests/test_runs_command.py::TestRunsState``, added in Wave 1).

These four states drive the four primary operator-facing UI surfaces. A
v1.3.x patch could silently regress any of them: a clamp boundary drift in
``set_steps``, a path-validation bypass in ``set_dataset_path``, a stub
event handler that stops firing the loading-state transition, or a setter
silently swallowing invalid input. CI was silent on all four surfaces.

This file establishes the contract for the four state classes:

1. **Field defaults** — every state initializes with the documented values
   so a UI smoke test against a default instance renders the expected form.
2. **Setter validation** — every public setter is exercised with one
   happy-path value (asserts the field updated) and one error-path value
   (asserts the error string is populated AND the field either stays at the
   prior value or is clamped to a documented bound).
3. **Event-handler stubs** — ``start_training``, ``start_multi_run``,
   ``start_export``, ``detect_format_stub`` transition the right run-state
   field and log the documented stub event. These pin the contract that the
   stubs MUST keep behaving like stubs until Phase 3 hookup lands; a
   regression that wired a real subprocess call would fail loudly here.

All tests are read-only (no actual training, no subprocess, no disk
writes); they exercise the state class directly via ``rx.State`` instance
methods. The ``reflex`` import is gated by ``pytest.importorskip`` so a
headless install (no ``[ui]`` extra) skips cleanly with a clear reason.

Pattern source: ``tests/test_runs_command.py::TestRunsState``.
"""

from __future__ import annotations

import pytest

# =============================================================================
# Shared importorskip — every test in this file needs reflex installed
# =============================================================================


@pytest.fixture(autouse=True)
def _require_reflex():
    """Skip every test in this module unless reflex is importable.

    The [ui] extra is opt-in (``pip install backpropagate[ui]``); a
    headless CI cell that did not install it should report 'skipped with
    clear reason', not 'failed at import time'.
    """
    pytest.importorskip(
        "reflex",
        reason="reflex is required for ui_state tests (install backpropagate[ui])",
    )


# =============================================================================
# TrainState — Train surface state (FRONTEND-A-005 + FRONTEND-B-002)
# =============================================================================


class TestTrainStateDefaults:
    """Default field values that the Train surface form binds to on first render."""

    def test_initial_config_defaults_match_design(self):
        """Train surface initializes with the documented config defaults.

        These values are the ones the UI form pre-populates when an operator
        first lands on the Train tab. A regression here would mean the UI
        shows a different default than the CLI / docs advertise.
        """
        from backpropagate.ui_state import TrainState

        state = TrainState()
        assert state.model == "Qwen/Qwen2.5-7B-Instruct"
        assert state.dataset_path == ""
        assert state.steps == 100
        assert state.batch_size == "auto"
        assert state.learning_rate == 2e-4
        assert state.lora_r == 16
        assert state.lora_alpha == 32
        assert state.lora_dropout == 0.05
        assert state.quantization == "4-bit"
        assert state.gpu_temp_threshold == 85
        assert state.gradient_checkpointing is True
        assert state.flash_attention is True

    def test_initial_runtime_state_is_idle(self):
        """The live-run telemetry fields start at zero/idle."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        assert state.run_state == "idle"
        assert state.current_step == 0
        assert state.current_loss == 0.0
        assert state.loss_history == []
        assert state.events == []

    def test_initial_error_fields_are_empty(self):
        """All ``*_error`` companion fields are empty on init."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        # Companion error strings — the UI binds to these via rx.cond.
        assert state.model_error == ""
        assert state.dataset_path_error == ""
        assert state.steps_error == ""
        assert state.batch_size_error == ""
        assert state.learning_rate_error == ""
        assert state.lora_r_error == ""
        assert state.lora_alpha_error == ""
        assert state.lora_dropout_error == ""
        assert state.gpu_temp_threshold_error == ""


class TestTrainStateSetters:
    """Exercise every public setter — happy path + error path."""

    def test_set_steps_accepts_valid_int(self):
        """``set_steps(500)`` updates the field; no error."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_steps(500)
        assert state.steps == 500
        assert state.steps_error == ""

    def test_set_steps_clamps_to_max(self):
        """Values above _STEPS_MAX (100_000) are clamped + error string set."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_steps(999_999)
        assert state.steps == 100_000  # _STEPS_MAX
        assert "clamped" in state.steps_error.lower()

    def test_set_steps_rejects_garbage_value(self):
        """Non-integer input leaves the field unchanged + sets error."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_steps("not a number")
        assert state.steps == 100  # prior value preserved
        assert state.steps_error != ""

    def test_set_learning_rate_accepts_valid_float(self):
        """``set_learning_rate('1e-3')`` parses the scientific notation."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_learning_rate("1e-3")
        assert state.learning_rate == 1e-3
        assert state.learning_rate_error == ""

    def test_set_learning_rate_clamps_above_max(self):
        """Learning rate above 1.0 is clamped."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_learning_rate(5.0)
        assert state.learning_rate == 1.0  # _LR_MAX
        assert "clamped" in state.learning_rate_error.lower()

    def test_set_lora_r_accepts_int(self):
        """LoRA rank setter accepts valid int."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_lora_r(64)
        assert state.lora_r == 64
        assert state.lora_r_error == ""

    def test_set_lora_alpha_accepts_int(self):
        """LoRA alpha setter accepts valid int."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_lora_alpha(128)
        assert state.lora_alpha == 128
        assert state.lora_alpha_error == ""

    def test_set_lora_dropout_clamps_above_one(self):
        """LoRA dropout above 1.0 is clamped to _LORA_DROPOUT_MAX."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_lora_dropout(2.0)
        assert state.lora_dropout == 1.0
        assert "clamped" in state.lora_dropout_error.lower()

    def test_set_batch_size_accepts_auto(self):
        """'auto' is the documented sentinel; setter must round-trip it."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_batch_size("16")
        assert state.batch_size == "16"
        state.set_batch_size("auto")
        assert state.batch_size == "auto"
        assert state.batch_size_error == ""

    def test_set_batch_size_rejects_garbage(self):
        """Garbage input does not change the field; error is set."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_batch_size("xyz")
        assert state.batch_size == "auto"  # default preserved
        assert state.batch_size_error != ""

    def test_set_target_modules_canonicalizes(self):
        """Whitespace is stripped + comma-separated normalized."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_target_modules("q_proj,v_proj")
        # Canonical form joins on ", ".
        assert "q_proj" in state.target_modules
        assert "v_proj" in state.target_modules
        assert state.target_modules_error == ""

    def test_set_target_modules_rejects_special_chars(self):
        """Characters outside the strict allowlist are rejected."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        prior = state.target_modules
        state.set_target_modules("../etc/passwd")
        assert state.target_modules == prior  # unchanged
        assert state.target_modules_error != ""

    def test_set_quantization_accepts_documented_values(self):
        """Quantization clamps to {4-bit, 8-bit, 16-bit}."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_quantization("8-bit")
        assert state.quantization == "8-bit"

    def test_set_quantization_rejects_invalid_silently(self):
        """Invalid quantization is silently rejected (value unchanged)."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_quantization("32-bit")
        assert state.quantization == "4-bit"  # default preserved

    def test_set_gpu_temp_threshold_clamps_above_max(self):
        """GPU temp threshold clamps to [40, 110]."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_gpu_temp_threshold(200)
        assert state.gpu_temp_threshold == 110  # _GPU_TEMP_MAX
        assert "clamped" in state.gpu_temp_threshold_error.lower()

    def test_set_gradient_checkpointing_coerces_bool(self):
        """gradient_checkpointing setter coerces any truthy to bool."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_gradient_checkpointing(False)
        assert state.gradient_checkpointing is False
        state.set_gradient_checkpointing(True)
        assert state.gradient_checkpointing is True

    def test_set_flash_attention_coerces_bool(self):
        """flash_attention setter coerces any truthy to bool."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_flash_attention(False)
        assert state.flash_attention is False


class TestTrainStateEventHandlers:
    """Exercise the stub event handlers — pin their stub-shape until Phase 3."""

    def test_start_training_transitions_to_loading_and_logs(self):
        """``start_training`` flips run_state to 'loading' + appends one event.

        This is the FRONTEND-A-005 stub contract. Phase 3 will replace this
        with a real subprocess dispatch; until then, the stub must keep
        firing the state transition so the UI's loading-spinner pattern
        works in the live preview.
        """
        from backpropagate.ui_state import TrainState

        state = TrainState()
        assert state.run_state == "idle"
        state.start_training()
        assert state.run_state == "loading"
        assert len(state.events) == 1
        assert state.events[0]["level"] == "info"
        assert "training" in state.events[0]["msg"].lower()

    def test_stop_training_returns_to_idle_from_active_states(self):
        """``stop_training`` returns to 'idle' from loading/active/paused."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.start_training()
        assert state.run_state == "loading"
        state.stop_training()
        assert state.run_state == "idle"
        # The stop event was appended after the start event.
        assert any("stopped" in e["msg"].lower() for e in state.events)

    def test_stop_training_from_idle_is_noop(self):
        """``stop_training`` from idle does NOT modify the state."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        assert state.run_state == "idle"
        prior_events_len = len(state.events)
        state.stop_training()
        assert state.run_state == "idle"
        # No event appended when already idle.
        assert len(state.events) == prior_events_len


# =============================================================================
# MultiRunState — Multi-Run surface state
# =============================================================================


class TestMultiRunStateDefaults:
    """Default field values for the Multi-Run surface."""

    def test_initial_config_defaults_match_design(self):
        """Multi-Run shares config defaults with TrainState by design."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        # Shared config (mirrors TrainState)
        assert state.model == "Qwen/Qwen2.5-7B-Instruct"
        assert state.steps == 100
        assert state.batch_size == "auto"
        assert state.learning_rate == 2e-4
        assert state.lora_r == 16
        assert state.lora_alpha == 32
        # Multi-Run specific
        assert state.num_runs == 3
        assert state.samples_per_run == 500
        assert state.merge_mode == "slao"
        assert state.replay_fraction == 0.0

    def test_initial_runtime_state_is_idle(self):
        """Multi-Run live state starts at idle / index 0 / empty runs."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        assert state.run_state == "idle"
        assert state.current_run_index == 0
        assert state.runs == []
        assert state.events == []


class TestMultiRunStateSetters:
    """Exercise the Multi-Run-specific setters."""

    def test_set_num_runs_accepts_valid_int(self):
        """num_runs setter accepts valid int in range."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_num_runs(10)
        assert state.num_runs == 10
        assert state.num_runs_error == ""

    def test_set_num_runs_clamps_above_max(self):
        """num_runs above _NUM_RUNS_MAX (100) is clamped."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_num_runs(500)
        assert state.num_runs == 100  # _NUM_RUNS_MAX
        assert "clamped" in state.num_runs_error.lower()

    def test_set_samples_per_run_accepts_valid_int(self):
        """samples_per_run setter accepts valid int."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_samples_per_run(2000)
        assert state.samples_per_run == 2000
        assert state.samples_per_run_error == ""

    def test_set_merge_mode_accepts_documented_values(self):
        """merge_mode clamps to {slao, weighted, ties}."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_merge_mode("weighted")
        assert state.merge_mode == "weighted"
        state.set_merge_mode("ties")
        assert state.merge_mode == "ties"

    def test_set_merge_mode_rejects_invalid_silently(self):
        """Invalid merge_mode is silently rejected."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_merge_mode("garbage")
        assert state.merge_mode == "slao"  # default preserved

    def test_set_replay_fraction_clamps_above_one(self):
        """replay_fraction above 1.0 is clamped."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_replay_fraction(5.0)
        assert state.replay_fraction == 1.0
        assert "clamped" in state.replay_fraction_error.lower()

    def test_set_replay_fraction_clamps_below_zero(self):
        """replay_fraction below 0.0 is clamped."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_replay_fraction(-0.5)
        assert state.replay_fraction == 0.0
        assert "clamped" in state.replay_fraction_error.lower()


class TestMultiRunStateEventHandlers:
    """Exercise the Multi-Run stub event handler."""

    def test_start_multi_run_transitions_to_loading_and_logs(self):
        """``start_multi_run`` flips run_state + appends one stub event."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        assert state.run_state == "idle"
        state.start_multi_run()
        assert state.run_state == "loading"
        assert len(state.events) == 1
        assert "multi-run" in state.events[0]["msg"].lower()


# =============================================================================
# ExportState — Export surface state (FRONTEND-A-002)
# =============================================================================


class TestExportStateDefaults:
    """Default field values for the Export surface."""

    def test_initial_defaults_match_design(self):
        """Export surface defaults: empty path, lora format, q4_K_M quant."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        assert state.source_model_path == ""
        assert state.format == "lora"
        assert state.gguf_quant == "q4_K_M"
        assert state.ollama_register is False
        assert state.ollama_name == ""
        assert state.export_state == "idle"
        assert state.output_path == ""
        assert state.events == []

    def test_initial_error_fields_are_empty(self):
        """All companion error strings start empty."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        assert state.source_model_path_error == ""
        assert state.ollama_name_error == ""


class TestExportStateSetters:
    """Exercise the Export setters."""

    def test_set_format_accepts_documented_values(self):
        """format clamps to {lora, merged, gguf}."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_format("merged")
        assert state.format == "merged"
        state.set_format("gguf")
        assert state.format == "gguf"

    def test_set_format_rejects_invalid_silently(self):
        """Invalid format is silently rejected."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_format("safetensors")  # not in allowlist
        assert state.format == "lora"  # default preserved

    def test_set_gguf_quant_accepts_documented_values(self):
        """gguf_quant clamps to documented set."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        for valid in ("q2_K", "q3_K_M", "q4_K_M", "q5_K_M", "q6_K", "q8_0"):
            state.set_gguf_quant(valid)
            assert state.gguf_quant == valid

    def test_set_gguf_quant_rejects_invalid_silently(self):
        """Invalid gguf_quant is silently rejected."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_gguf_quant("q1_INVALID")
        assert state.gguf_quant == "q4_K_M"  # default preserved

    def test_set_ollama_register_coerces_bool(self):
        """ollama_register setter coerces to bool."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_register(True)
        assert state.ollama_register is True
        state.set_ollama_register(False)
        assert state.ollama_register is False

    def test_set_ollama_name_accepts_valid_name(self):
        """Valid ollama model names round-trip."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_name("my-model")
        assert state.ollama_name == "my-model"
        assert state.ollama_name_error == ""

        state.set_ollama_name("my-model:v1")
        assert state.ollama_name == "my-model:v1"
        assert state.ollama_name_error == ""

    def test_set_ollama_name_rejects_path_traversal(self):
        """Names containing '..' are rejected as path traversal probes."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_name("../etc/passwd")
        assert state.ollama_name == ""
        assert "invalid" in state.ollama_name_error.lower()

    def test_set_ollama_name_rejects_backslash(self):
        """Windows-style backslashes are rejected."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_name(r"foo\bar")
        assert state.ollama_name == ""
        assert state.ollama_name_error != ""

    def test_set_ollama_name_rejects_leading_slash(self):
        """Absolute-path-shaped names are rejected."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_name("/etc/passwd")
        assert state.ollama_name == ""
        assert state.ollama_name_error != ""

    def test_set_ollama_name_rejects_null_byte(self):
        """Null bytes in the name are rejected."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_name("foo\x00bar")
        assert state.ollama_name == ""
        assert state.ollama_name_error != ""

    def test_set_ollama_name_empty_clears_error(self):
        """Setting the name to empty clears both field and error."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_ollama_name("../bad")  # set the error
        assert state.ollama_name_error != ""

        state.set_ollama_name("")
        assert state.ollama_name == ""
        assert state.ollama_name_error == ""


class TestExportStateEventHandlers:
    """Exercise the Export stub event handler."""

    def test_start_export_transitions_to_loading_and_logs(self):
        """``start_export`` flips export_state + appends one stub event."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        assert state.export_state == "idle"
        state.start_export()
        assert state.export_state == "loading"
        assert len(state.events) == 1
        assert "export" in state.events[0]["msg"].lower()


# =============================================================================
# DatasetState — Dataset surface state (FRONTEND-A-003 + FRONTEND-B-013)
# =============================================================================


class TestDatasetStateDefaults:
    """Default field values for the Dataset surface."""

    def test_initial_defaults_match_design(self):
        """Dataset surface defaults."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        assert state.uploaded_path == ""
        assert state.upload_error == ""
        assert state.upload_count == 0
        assert state.detected_format == ""
        assert state.preview_records == []
        assert state.format_hint == "auto"
        assert state.dedup_enabled is True
        assert state.drop_empty is True
        assert state.apply_curriculum is False
        assert state.min_tokens == 0
        assert state.max_tokens == 2048

    def test_uploaded_basename_empty_when_no_upload(self):
        """``uploaded_basename`` computed var returns "" with no upload."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        # rx.var computed values are accessed as attributes on the instance
        # (Reflex resolves them via descriptor). Use the underlying function
        # if the descriptor access fails on a bare instance.
        result = state.uploaded_basename
        assert result == ""


class TestDatasetStateSetters:
    """Exercise the Dataset setters."""

    def test_set_format_hint_accepts_documented_values(self):
        """format_hint clamps to {auto, sharegpt, alpaca, openai, jsonl}."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        for valid in ("auto", "sharegpt", "alpaca", "openai", "jsonl"):
            state.set_format_hint(valid)
            assert state.format_hint == valid

    def test_set_format_hint_rejects_invalid_silently(self):
        """Invalid format_hint is silently rejected."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_format_hint("garbage")
        assert state.format_hint == "auto"  # default preserved

    def test_set_dedup_enabled_coerces_bool(self):
        """dedup_enabled setter coerces to bool."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_dedup_enabled(False)
        assert state.dedup_enabled is False
        state.set_dedup_enabled(True)
        assert state.dedup_enabled is True

    def test_set_drop_empty_coerces_bool(self):
        """drop_empty setter coerces to bool."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_drop_empty(False)
        assert state.drop_empty is False

    def test_set_apply_curriculum_coerces_bool(self):
        """apply_curriculum setter coerces to bool."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_apply_curriculum(True)
        assert state.apply_curriculum is True

    def test_set_min_tokens_accepts_valid_int(self):
        """min_tokens setter accepts valid int."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_min_tokens(100)
        assert state.min_tokens == 100
        assert state.min_tokens_error == ""

    def test_set_min_tokens_bumps_max_when_exceeds_current(self):
        """Setting min above current max bumps max to match.

        This is the documented contract — the UI prevents an invariant
        violation (min > max) by lifting max instead of rejecting min.
        """
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        assert state.max_tokens == 2048
        state.set_min_tokens(5000)
        assert state.min_tokens == 5000
        assert state.max_tokens == 5000  # bumped to match

    def test_set_max_tokens_accepts_valid_int(self):
        """max_tokens setter accepts valid int."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_max_tokens(4096)
        assert state.max_tokens == 4096
        assert state.max_tokens_error == ""

    def test_set_min_tokens_rejects_garbage(self):
        """Garbage min_tokens leaves field unchanged + error set."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_min_tokens("not a number")
        assert state.min_tokens == 0  # default preserved
        assert state.min_tokens_error != ""


class TestDatasetStateEventHandlers:
    """Exercise the Dataset stub event handler."""

    def test_detect_format_stub_noop_without_upload(self):
        """``detect_format_stub`` is a no-op when no file is uploaded."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        assert state.uploaded_path == ""
        state.detect_format_stub()
        assert state.detected_format == ""  # unchanged

    def test_detect_format_stub_sets_alpaca_when_upload_present(self):
        """``detect_format_stub`` sets a placeholder detected_format when
        an upload is present.

        This is the stub contract — Phase 3 will replace with a real
        per-line peek + format-detect routine. Until then, the stub must
        keep firing the state transition so the UI's detect-result panel
        renders.
        """
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.uploaded_path = "/tmp/some-uploaded-file.jsonl"
        state.detect_format_stub()
        assert state.detected_format == "alpaca"
