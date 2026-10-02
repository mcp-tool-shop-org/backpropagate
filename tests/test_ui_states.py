"""Smoke tests for the four v1.1.0-shipped Reflex UI state classes.

TESTS-B-007 (v1.3 swarm Stage C humanization): The FRONTEND-A-002 /
FRONTEND-A-005 / FRONTEND-A-003 / FRONTEND-B-013 Reflex state classes —
``TrainState``, ``MultiRunState``, ``ExportState``, ``DatasetState`` — had
ZERO regression test coverage. Only ``RunsState`` was covered (by
``tests/test_runs_command.py::TestRunsState``, added in Wave 1).

These four states drive the four primary operator-facing UI surfaces. A
v1.3.x patch could silently regress any of them: a clamp boundary drift in
``set_steps``, a path-validation bypass in ``set_dataset_path``, a Start
handler that re-grows a fake loading spinner, or a setter silently
swallowing invalid input. CI was silent on all four surfaces.

This file establishes the contract for the four state classes:

1. **Field defaults** — every state initializes with the documented values
   so a UI smoke test against a default instance renders the expected form.
2. **Setter validation** — every public setter is exercised with one
   happy-path value (asserts the field updated) and one error-path value
   (asserts the error string is populated AND the field either stays at the
   prior value or is clamped to a documented bound).
3. **Start handlers** — since ui-v2 P1/P2 ``start_training``,
   ``start_multi_run`` and ``start_export`` start real jobs through the
   JobManager (a child process; never training in the server). The
   multi-run and export pages hand their job to ``TrainState.start_job``. These
   pin the honesty contract; a regression that re-grew the fake loading
   spinner — or, worse, wired a real subprocess call here — fails loudly.

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
        # ui-v2 P3 (resolves TESTS-A-006): the form opens on the CLI's own
        # defaults, the v1.3 "quality" LoRA shape (r 256, alpha 512, all
        # linear layers) on a 4-bit base with SFT.
        assert state.preset == "qwen2.5-7b"
        assert state.lora_r == 256
        assert state.lora_alpha == 512
        assert state.lora_dropout == 0.05
        assert state.target_modules == "all-linear"
        assert state.train_mode == "qlora"
        assert state.method == "sft"
        assert state.gpu_temp_threshold == 90
        assert state.gradient_checkpointing is True

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

    def test_set_train_mode_accepts_documented_values(self):
        """ui-v2 P3: the mode picker is qlora | lora | full."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_train_mode("lora")
        assert state.train_mode == "lora"

    def test_set_train_mode_rejects_invalid_silently(self):
        """Invalid mode is silently rejected (value unchanged)."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_train_mode("8-bit")
        assert state.train_mode == "qlora"  # default preserved

    def test_set_gpu_temp_threshold_clamps_above_max(self):
        """GPU temp threshold clamps to the CLI's --gpu-max-temp range [50, 105]."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_gpu_temp_threshold(200)
        assert state.gpu_temp_threshold == 105  # _GPU_TEMP_MAX
        assert "clamped" in state.gpu_temp_threshold_error.lower()

    def test_set_gradient_checkpointing_coerces_bool(self):
        """gradient_checkpointing setter coerces any truthy to bool."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.set_gradient_checkpointing(False)
        assert state.gradient_checkpointing is False
        state.set_gradient_checkpointing(True)
        assert state.gradient_checkpointing is True

    def test_no_flash_attention_toggle(self):
        """ui-v2 P3: the trainer has no flash-attention knob, so the form has
        no checkbox that would do nothing."""
        from backpropagate.ui_state import TrainState

        assert not hasattr(TrainState, "set_flash_attention")


class TestTrainStateEventHandlers:
    """Exercise the Start/Stop event handlers.

    ui-v2 P1: ``start_training`` spawns a real training subprocess via
    ``ui_jobs.JobManager`` (the CLIUI-B-002 hookup, now landed). The honesty
    floor survives as the REFUSAL contract: anything that cannot start
    (invalid form, dataset outside the sandbox, second concurrent run) sets
    ``job_refusal`` on screen and never touches a fake loading state.
    """

    def test_start_training_refuses_without_dataset(self, tmp_path):
        """No dataset -> refusal on screen, run_state never leaves idle."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        assert state.run_state == "idle"
        assert state.job_refusal == ""
        state.start_training()
        assert state.run_state == "idle"
        assert state.job_refusal != ""
        assert state.job_id == ""

    def test_start_training_launches_child_process_when_valid(
        self, tmp_path, monkeypatch
    ):
        """A valid spec spawns the child and turns the rail live."""
        import backpropagate.ui_jobs as ui_jobs
        import backpropagate.ui_security as sec
        from backpropagate.ui_state import TrainState

        monkeypatch.setattr(sec, "get_ui_output_dir", lambda: tmp_path)
        data = tmp_path / "ds.jsonl"
        data.write_text('{"text": "hi"}\n')

        class _FakeManager:
            def __init__(self):
                self.specs = []

            def start(self, spec):
                self.specs.append(spec)
                return ui_jobs.JobHandle(
                    job_id="run_20261001_000000_1_ab12",
                    run_dir=tmp_path / "run_20261001_000000_1_ab12",
                    pid=4242,
                    started_at=0.0,
                    spec=spec,
                )

        fake = _FakeManager()
        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: fake)

        state = TrainState()
        state.dataset_path = str(data)
        state.start_training()
        assert state.run_state == "active"
        assert state.job_id.startswith("run_")
        assert state.job_refusal == ""
        assert fake.specs and fake.specs[0].kind == "sft"
        assert fake.specs[0].steps == state.steps
        assert any("started" in e["msg"] for e in state.events)

    def test_start_training_refusal_when_manager_busy(self, tmp_path, monkeypatch):
        """A second concurrent start is refused ON SCREEN (handoff rule)."""
        import backpropagate.ui_jobs as ui_jobs
        import backpropagate.ui_security as sec
        from backpropagate.ui_state import TrainState

        monkeypatch.setattr(sec, "get_ui_output_dir", lambda: tmp_path)
        data = tmp_path / "ds.jsonl"
        data.write_text('{"text": "hi"}\n')

        class _BusyManager:
            def start(self, spec):
                raise ui_jobs.JobRefusedError("A job is already running (run_x).")

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _BusyManager())
        state = TrainState()
        state.dataset_path = str(data)
        state.start_training()
        assert state.run_state == "idle"
        assert "already running" in state.job_refusal

    def test_apply_job_event_folds_step_and_terminal(self):
        """The poller's row-fold drives step/loss/VRAM/temp and terminal state."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.job_id = "run_x"
        state.run_state = "active"
        state._apply_job_event(
            {
                "kind": "step",
                "step": 12,
                "total_steps": 40,
                "loss": 0.42,
                "vram_reserved_gib": 9.5,
                "temp_c": 61.0,
            }
        )
        assert state.current_step == 12
        assert state.job_total_steps == 40
        assert state.current_loss == 0.42
        assert state.loss_history == [0.42]
        assert state.step_history == [12]
        assert state.vram_used_gb == 9.5
        assert state.gpu_temp == 61.0
        # device-wide VRAM keys win over the per-process fallback when present
        state._apply_job_event(
            {
                "kind": "step",
                "step": 13,
                "total_steps": 40,
                "loss": 0.40,
                "ema_loss": 0.41,
                "vram_device_used_gib": 12.0,
                "vram_device_total_gib": 31.8,
                "vram_reserved_gib": 9.6,
            }
        )
        assert state.vram_used_gb == 12.0
        assert state.vram_total_gb == 31.8
        assert state.step_history == [12, 13]
        assert state.ema_history[-1] == 0.41
        assert state.loss_label == "0.4000"
        assert state.ema_loss_label == "0.4100"
        state._apply_job_event(
            {"kind": "error", "code": "RUNTIME_OOM", "message": "cuda oom", "hint": "smaller batch"}
        )
        assert state.job_error_code == "RUNTIME_OOM"
        assert state.job_error_hint == "smaller batch"
        state._finalize_job({"status": "stopped", "output_path": "out/run_x"})
        assert state.run_state == "stopped"
        assert state.job_output_path == "out/run_x"
        assert any("Stopped early" in e["msg"] for e in state.events)
        state._finalize_job({"status": "crashed"})
        assert state.run_state == "error"

    def test_stop_training_idles_when_no_active_job(self, monkeypatch):
        """Stop with nothing running is a no-op landing on idle."""
        import backpropagate.ui_jobs as ui_jobs
        from backpropagate.ui_state import TrainState

        class _IdleManager:
            def is_alive(self, job_id):
                return False

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _IdleManager())
        state = TrainState()
        state.job_id = "run_nothing"
        state.run_state = "active"
        state.stop_training()
        assert state.run_state == "idle"

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
        # ui-v2 P3: same CLI defaults as TrainState (quality LoRA shape).
        assert state.lora_r == 256
        assert state.lora_alpha == 512
        assert state.train_mode == "qlora"
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
        """merge_mode clamps to {slao, simple, ties} (the CLI's choices)."""
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        state.set_merge_mode("simple")
        assert state.merge_mode == "simple"
        state.set_merge_mode("weighted")  # not a CLI merge mode
        assert state.merge_mode == "simple"
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
    """The Multi-Run Start button hands a job to the shared job state (P2)."""

    def test_start_multi_run_returns_a_start_job_event_and_fakes_nothing(self):
        from backpropagate.ui_state import MultiRunState

        state = MultiRunState()
        spec = state.start_multi_run()
        assert spec.handler.fn.__name__ in ("start_job", "refuse")
        # No local fake progress: the job state (TrainState) owns the run.
        assert state.run_state == "idle"
        assert state.events == []


# =============================================================================
# ExportState — Export surface state (FRONTEND-A-002)
# =============================================================================


class TestExportStateDefaults:
    """Default field values for the Export surface."""

    def test_initial_defaults_match_design(self):
        """Export surface defaults: empty path, lora format, q4_k_m quant."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        assert state.source_model_path == ""
        assert state.format == "lora"
        assert state.gguf_quant == "q4_k_m"
        assert state.ollama_register is False
        assert state.ollama_name == ""
        assert state.export_state == "idle"
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
        for valid in ("f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k"):
            state.set_gguf_quant(valid)
            assert state.gguf_quant == valid

    def test_set_gguf_quant_rejects_invalid_silently(self):
        """Invalid gguf_quant is silently rejected."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_gguf_quant("q1_INVALID")
        assert state.gguf_quant == "q4_k_m"  # default preserved

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


class TestExportStateHubTokenFilePath:
    """The token-file field accepts a file inside ``~/.backpropagate/``.

    The page shows the file name. The resolved path stays on the backend
    var. A file outside that folder, including ``~/.config``, is refused
    and the error text does not contain the path.
    """

    def test_file_inside_backpropagate_is_accepted_by_name(self, tmp_path, monkeypatch):
        from pathlib import Path

        from backpropagate.ui_state import ExportState

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        token_file = home / ".backpropagate" / "hf-token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("hf_" + "a" * 40, encoding="utf-8")

        state = ExportState()
        state.set_hub_token_file_path(str(token_file))
        assert state.hub_token_file_path_error == ""
        assert state.hub_token_file_path == "hf-token"
        assert state._hub_token_file_path == str(token_file.resolve())
        config_file = home / ".config" / "backpropagate" / "hf-token"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("hf_" + "b" * 40, encoding="utf-8")
        state.set_hub_token_file_path(str(config_file))
        assert state.hub_token_file_path == ""
        assert str(config_file.resolve()) not in state.hub_token_file_path_error

    def test_nonexistent_path_is_rejected(self, tmp_path):
        """A path that does not exist is rejected with a clear error."""
        from backpropagate.ui_state import ExportState

        missing = tmp_path / "does-not-exist" / "hf-token"
        state = ExportState()
        state.set_hub_token_file_path(str(missing))
        assert state.hub_token_file_path == "", (
            "a nonexistent token file must not be staged"
        )
        assert state.hub_token_file_path_error != "", (
            "a nonexistent token file must surface an operator-facing error"
        )

    def test_directory_is_rejected(self, tmp_path):
        """A directory (not a regular file) is rejected with a clear error."""
        from backpropagate.ui_state import ExportState

        a_dir = tmp_path / "some-dir"
        a_dir.mkdir()
        state = ExportState()
        state.set_hub_token_file_path(str(a_dir))
        assert state.hub_token_file_path == "", (
            "a directory must not be staged as a token file"
        )
        assert state.hub_token_file_path_error != "", (
            "a directory must surface an operator-facing error"
        )
        assert "file" in state.hub_token_file_path_error.lower(), (
            "the directory-rejection error should mention it expected a file; "
            f"got {state.hub_token_file_path_error!r}"
        )

    def test_null_byte_is_rejected(self, tmp_path):
        """A NUL byte in the path is rejected (hostile input)."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_hub_token_file_path(str(tmp_path / "hf\x00token"))
        assert state.hub_token_file_path == ""
        assert state.hub_token_file_path_error != ""

    def test_empty_clears_field_and_error(self, tmp_path):
        """Empty input clears both the path and the error (pass-through)."""
        from backpropagate.ui_state import ExportState

        state = ExportState()
        # First set an error.
        state.set_hub_token_file_path(str(tmp_path / "missing"))
        assert state.hub_token_file_path_error != ""
        # Now clear it.
        state.set_hub_token_file_path("")
        assert state.hub_token_file_path == ""
        assert state.hub_token_file_path_error == ""


class TestExportStateEventHandlers:
    """The Export button hands a job to the shared job state (P2)."""

    def test_start_export_returns_a_start_job_event_and_fakes_nothing(self):
        from backpropagate.ui_state import ExportState

        state = ExportState()
        spec = state.start_export()
        assert spec.handler.fn.__name__ == "start_job"
        assert state.export_state == "idle"
        assert state.events == []


# =============================================================================
# DatasetState — Dataset surface state (FRONTEND-A-003 + FRONTEND-B-013)
# =============================================================================


class TestDatasetStateDefaults:
    """Default field values for the Dataset surface."""

    def test_initial_defaults_match_design(self):
        """Dataset surface defaults."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        # UI-A-002 (Wave A2): the full upload path is now a backend-only var
        # (``_uploaded_path``); the public surface is the basename + a boolean.
        assert state._uploaded_path == ""
        assert state.uploaded_basename == ""
        assert state.has_upload is False
        assert state.upload_error == ""
        assert state.upload_count == 0
        assert state.detected_format == ""
        assert state.preview_records == []
        assert state.format_hint == "auto"
        assert state.dedup_enabled is True
        assert state.drop_empty is True
        assert state.apply_curriculum is False
        assert state.min_tokens == 0
        # 0 = no upper limit: nothing is removed for length until asked.
        assert state.max_tokens == 0

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
        state.set_max_tokens(2048)
        state.set_min_tokens(5000)
        assert state.min_tokens == 5000
        assert state.max_tokens == 5000  # bumped to match

    def test_set_min_tokens_leaves_no_limit_alone(self):
        """A longest-length of 0 means "no limit" and is never bumped."""
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_min_tokens(5000)
        assert (state.min_tokens, state.max_tokens) == (5000, 0)

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
    """The placeholder format handler is gone: the format comes from the file."""

    def test_no_placeholder_handler_remains(self):
        from backpropagate.ui_state import DatasetState

        assert not hasattr(DatasetState, "detect_format_stub")

    def test_a_format_is_never_claimed_without_reading_a_file(self):
        from backpropagate.ui_state import DatasetState

        state = DatasetState()
        state.set_format_hint("alpaca")
        assert state.detected_format == ""  # the hint is a setting, not a finding


# =============================================================================
# AppState — theme toggle + FRONTEND-F-001 (Wave 5.5) theme-token contract
# =============================================================================


class TestAppStateTheme:
    """The theme-toggle state contract.

    FRONTEND-F-001 (Wave 5.5): the v1.2 bug was that ``AppState.toggle_theme``
    flipped a server-side bool but no DOM mutation reached the document
    root, so the LIGHT_TOKENS palette never activated. The fix moves the
    load-bearing toggle to Reflex's built-in ``rx.color_mode`` +
    ``rx.toggle_color_mode`` (DOM-mutating, prefers-color-scheme aware).

    These tests pin two contracts:

    1. The legacy ``AppState.theme`` field + ``toggle_theme`` event still
       work as before (back-compat for any external automation that reads
       state directly).
    2. The TOKENS_CSS stylesheet keys off the selectors that Reflex's
       next-themes provider actually emits (``.light`` / ``.light-theme``)
       plus the legacy ``[data-theme="light"]`` fallback — the v1.2 bug
       was that ONLY the last selector was emitted and Reflex never set
       it.
    """

    def test_initial_theme_is_dark(self):
        """Default theme field is ``dark`` (legacy back-compat)."""
        from backpropagate.ui_state import AppState

        state = AppState()
        assert state.theme == "dark"

    def test_toggle_theme_flips_state(self):
        """Legacy ``toggle_theme`` event still flips dark<->light.

        Kept for back-compat with any external automation reading the
        state field directly; the load-bearing DOM update happens via
        ``rx.toggle_color_mode`` wired into ``BpHeader``.
        """
        from backpropagate.ui_state import AppState

        state = AppState()
        assert state.theme == "dark"
        state.toggle_theme()
        assert state.theme == "light"
        state.toggle_theme()
        assert state.theme == "dark"

    def test_tokens_css_emits_light_class_selectors(self):
        """The LIGHT_TOKENS block fires on ``.light`` / ``.light-theme``.

        FRONTEND-F-001: Reflex's color-mode provider writes
        ``class="light"`` onto ``<html>`` (next-themes ``attribute="class"``).
        The v1.2 ``[data-theme="light"]`` selector never matched because
        nothing in the codebase set that attribute. This test pins the
        triple selector so future refactors can't silently regress.
        """
        from backpropagate.ui_theme import TOKENS_CSS

        # All three selectors must be present so the LIGHT_TOKENS block
        # fires regardless of which Reflex / Radix / next-themes shape is
        # active. Verify ordering is alphabetical-ish (consistent with the
        # ui_theme.py source).
        assert ".light" in TOKENS_CSS
        assert ".light-theme" in TOKENS_CSS
        assert '[data-theme="light"]' in TOKENS_CSS

    def test_tokens_css_emits_full_light_token_set(self):
        """The light-mode token set is present in the emitted CSS.

        Pin that a representative bg + text + accent token from
        LIGHT_TOKENS appears in the assembled stylesheet — guards against
        accidental drift between LIGHT_TOKENS dict and TOKENS_CSS.
        """
        from backpropagate.ui_theme import LIGHT_TOKENS, TOKENS_CSS

        # Hit the dominant surface + text + accent triplet.
        for key in ("--bp-bg", "--bp-text", "--bp-teal"):
            assert LIGHT_TOKENS[key] in TOKENS_CSS, (
                f"LIGHT_TOKENS[{key!r}] = {LIGHT_TOKENS[key]!r} missing from "
                "TOKENS_CSS — the LIGHT_TOKENS block was not emitted, which "
                "would re-strand the theme toggle"
            )

    def test_radix_theme_omits_hardcoded_appearance(self):
        """RADIX_THEME must NOT lock ``appearance`` — that goes at the
        rx.theme() call site as ``"inherit"`` (follow the <html> colour-mode class).

        FRONTEND-F-001: hard-coding ``appearance="dark"`` here was the
        v1.2 bug — it overrode the per-render binding and the toggle
        button became a no-op. This test pins the fix.
        """
        from backpropagate.ui_theme import RADIX_THEME

        assert "appearance" not in RADIX_THEME, (
            "RADIX_THEME contains a hard-coded 'appearance' — that strands "
            "the theme toggle button (the v1.2 FRONTEND-F-001 bug). The "
            "appearance must be set at the rx.theme() call site in "
            "ui_app/app.py (\"inherit\", so the <html> colour-mode class "
            "written by Reflex's provider takes effect)."
        )


# =============================================================================
# TESTS-F-004 (v1.4 Wave 6b): RunDetailState — was_deleted +
# action_in_flight pinning.
# =============================================================================
#
# Wave 3.5 FRONTEND-B-001 introduced ``was_deleted`` (a deletion-specific
# success surface separate from ``not_found``). Pre-fix, a successful
# ``delete_run`` shell-out set ``not_found = True`` — latching the page
# into the "Run not found" chrome (designed for unknown-id navigations)
# and stranding the success message in ``action_result`` behind a template
# that never displays it.
#
# Stage C FRONTEND-B-014-EXTENDED introduced ``action_in_flight`` for the
# diff/replay/delete/export shell-outs. Pre-fix the operator clicked the
# button and saw NO feedback until the subprocess returned — a long silent
# gap that read as a frozen UI. The handlers now set ``action_in_flight``
# at the start and clear it via try/finally so a thrown exception still
# leaves the UI in a consistent state.
#
# These pin: defaults, delete_run success/failure mutations,
# action_in_flight presence + always-cleared invariant, and load_run's
# was_deleted reset on remount.
#
# All tests stub ``shutil.which`` + ``subprocess.run`` so no real CLI is
# invoked.


class TestRunDetailStateDefaults:
    """Default field values for the per-run drill-down state."""

    def test_initial_was_deleted_false(self):
        """Wave 3.5 FRONTEND-B-001: ``was_deleted`` defaults to False so
        the deletion-confirmation chrome only renders after a successful
        delete_run handler.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        assert hasattr(state, "was_deleted"), (
            "Wave 3.5 FRONTEND-B-001 regression: RunDetailState is "
            "missing the was_deleted field. The template's branch on "
            "was_deleted BEFORE not_found has nothing to read."
        )
        assert state.was_deleted is False, (
            f"FRONTEND-B-001: RunDetailState.was_deleted must default "
            f"to False; got {state.was_deleted!r}. A True default "
            f"would render the 'Run deleted.' chrome unconditionally."
        )

    def test_initial_not_found_false(self):
        """``not_found`` defaults to False — separate surface from
        ``was_deleted`` per FRONTEND-B-001 disambiguation.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        assert state.not_found is False

    def test_initial_action_in_flight_empty(self):
        """Stage C FRONTEND-B-014-EXTENDED: ``action_in_flight``
        defaults to empty string (no shell-out active on first render).
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        assert hasattr(state, "action_in_flight"), (
            "Stage C FRONTEND-B-014-EXTENDED regression: "
            "RunDetailState is missing the action_in_flight field. "
            "The action panel's spinner / 'Running ...' chrome has "
            "nothing to bind to."
        )
        assert state.action_in_flight == "", (
            f"action_in_flight must default to '' (no active shell-out); "
            f"got {state.action_in_flight!r}."
        )

    def test_initial_action_result_and_error_empty(self):
        """Both action-panel surfaces start empty."""
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        assert state.action_result == ""
        assert state.action_error == ""


class TestRunDetailStateDeleteRun:
    """Wave 3.5 FRONTEND-B-001 + Stage C FRONTEND-B-014-EXTENDED: delete
    handler mutations.
    """

    def _make_subprocess_mock(self, *, returncode: int, stdout: str = "", stderr: str = ""):
        """Build a ``subprocess.run``-like return value."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    def _patch_in_process_delete(self, monkeypatch, tmp_path, *, ok: bool):
        """Patch RunHistoryManager + sandbox dir for the in-process delete.

        UI-A-004 (Wave A1): delete_run no longer shells out — it calls
        ``RunHistoryManager.delete_run`` in-process (as ``load_run`` does).
        These tests patch that surface instead of ``subprocess.run``.
        """
        from backpropagate import checkpoints as _ck

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                return ok

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

    def test_delete_run_success_sets_was_deleted_true(self, monkeypatch, tmp_path):
        """A successful in-process ``delete_run`` sets ``was_deleted=True``
        AND ``action_result`` (the deletion-confirmation chrome) AND
        clears ``action_error``.

        Pins the post-fix contract: the page renders 'Run deleted.' with
        a Back-to-runs affordance, NOT the not-found chrome.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "abc12345"

        self._patch_in_process_delete(monkeypatch, tmp_path, ok=True)

        state.delete_run()

        assert state.was_deleted is True, (
            "Wave 3.5 FRONTEND-B-001 regression: a successful "
            "delete_run shell-out must set was_deleted=True so the "
            "template renders the deletion-confirmation chrome."
        )
        assert state.action_error == "", (
            "Successful delete_run must clear action_error so a "
            "stale prior error doesn't render alongside the success."
        )
        assert "deleted" in state.action_result.lower(), (
            f"Successful delete_run must populate action_result with a "
            f"human-readable confirmation; got {state.action_result!r}."
        )
        # FRONTEND-B-001 explicitly pins the not_found surface STAYS
        # False on a successful delete (the chrome branches on
        # was_deleted BEFORE not_found, so a stale True here would
        # render the wrong chrome).
        assert state.not_found is False, (
            "FRONTEND-B-001 regression: successful delete_run set "
            "not_found=True. Pre-fix that latched the page into "
            "the 'Run not found' chrome and stranded the success "
            "message. The fix MUST keep not_found=False."
        )

    def test_delete_run_failure_does_not_set_was_deleted(self, monkeypatch, tmp_path):
        """An in-process delete that returns False (entry not found) leaves
        ``was_deleted=False`` so the not-found / error chrome can render
        correctly.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "abc12345"

        self._patch_in_process_delete(monkeypatch, tmp_path, ok=False)

        state.delete_run()

        assert state.was_deleted is False, (
            "Failed delete_run must NOT set was_deleted=True. The "
            "deletion-confirmation chrome only renders on success."
        )
        assert state.action_error != "", (
            "Failed delete_run must populate action_error so the "
            "operator sees the failure surface."
        )

    def test_load_run_resets_was_deleted_on_remount(self, monkeypatch, tmp_path):
        """``load_run`` clears ``was_deleted`` so navigating from a
        just-deleted run's URL to a different run id doesn't carry the
        'Run deleted.' surface forward.

        Pins the FRONTEND-B-001 remount-clear contract documented at
        ui_state.py:1653-1657.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.was_deleted = True  # simulate post-delete state
        state.current_run_id = "newrunid"

        # load_run reaches into the filesystem via get_ui_output_dir →
        # RunHistoryManager; we don't need a real run record because we
        # only care that was_deleted gets reset early in the handler.
        # Force the "no history directory" early-return so the handler
        # short-circuits cleanly after the was_deleted reset.
        nonexistent = tmp_path / "no-history-dir"
        # Don't create it — load_run takes the `if not history_dir.exists()`
        # branch and returns early, but ONLY after was_deleted = False.
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: nonexistent,
        )

        state.load_run()

        assert state.was_deleted is False, (
            "FRONTEND-B-001 regression: load_run did NOT reset "
            "was_deleted on remount. Navigating from a just-deleted "
            "run's URL to a different run id must not carry the "
            "deletion-confirmation chrome forward. See ui_state.py:1657."
        )


class TestRunDetailStateActionInFlight:
    """Stage C FRONTEND-B-014-EXTENDED: action_in_flight + try/finally
    clear contract.
    """

    def _make_subprocess_mock(self, *, returncode: int, stdout: str = "", stderr: str = ""):
        from unittest.mock import MagicMock

        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    def _patch_in_process_delete(self, monkeypatch, tmp_path, *, ok: bool):
        """Patch the in-process delete surface (UI-A-004) — see
        TestRunDetailStateDeleteRun._patch_in_process_delete.
        """
        from backpropagate import checkpoints as _ck

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                return ok

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

    def test_action_in_flight_cleared_after_successful_delete(self, monkeypatch, tmp_path):
        """After a successful in-process delete the field is cleared
        (via try/finally) so the spinner stops rendering.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "rid-delete"

        self._patch_in_process_delete(monkeypatch, tmp_path, ok=True)

        state.delete_run()

        assert state.action_in_flight == "", (
            "Stage C FRONTEND-B-014-EXTENDED regression: "
            "action_in_flight not cleared after a successful "
            "in-process delete_run. The try/finally MUST clear the "
            "field on every exit path so the spinner stops rendering."
        )

    def test_action_in_flight_cleared_after_failed_delete(self, monkeypatch, tmp_path):
        """After an in-process delete that returns False the field is STILL
        cleared via try/finally. Pre-fix the failure path could leave the
        spinner stuck.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "rid-delete-fail"

        self._patch_in_process_delete(monkeypatch, tmp_path, ok=False)

        state.delete_run()

        assert state.action_in_flight == "", (
            "FRONTEND-B-014-EXTENDED regression: action_in_flight "
            "not cleared after a FAILED in-process delete_run. The "
            "try/finally clear must fire on every exit path, "
            "including the not-found branch."
        )

    def test_action_in_flight_cleared_when_delete_raises(self, monkeypatch, tmp_path):
        """If the in-process delete raises (e.g. OSError from the history
        store), the finally MUST still clear ``action_in_flight``.

        UI-A-004 (Wave A1): delete_run is in-process now; the try/finally
        contract still holds — an exception must not latch the spinner.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "rid-raise"

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                raise OSError("disk gone")

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

        state.delete_run()  # handler must swallow the error

        assert state.action_in_flight == "", (
            "FRONTEND-B-014-EXTENDED regression: action_in_flight "
            "not cleared when the in-process delete raised. The "
            "try/finally MUST fire on the exception path so the "
            "spinner doesn't latch forever."
        )
        assert state.action_error != "", (
            "Exception path must populate action_error so the "
            "operator sees what went wrong."
        )

    def test_action_in_flight_cleared_after_diff_against(self, monkeypatch):
        """The same try/finally contract holds for diff_against
        (sibling handler at ui_state.py:1832).
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "rid-diff-a"

        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/local/bin/backprop" if name in ("backprop", "backpropagate") else None,
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: self._make_subprocess_mock(
                returncode=0, stdout="diff output"
            ),
        )

        state.diff_against("rid-diff-b")

        assert state.action_in_flight == "", (
            "FRONTEND-B-014-EXTENDED regression on diff_against: "
            "action_in_flight not cleared after successful diff-runs "
            "shell-out. The grep-all-instances doctrine "
            "[[grep-all-instances-when-fixing-pattern]] requires this "
            "contract to hold across all 4 action handlers "
            "(diff/replay/delete/export)."
        )

    def test_action_in_flight_cleared_after_replay(self, monkeypatch, tmp_path):
        """The same try/finally contract holds for the in-process replay
        preflight.

        replay was rewired off the phantom ``replay --dry-run`` shell-out to
        an in-process RunHistoryManager check (see
        TestRunDetailReplayInProcess); the try/finally clear contract still
        applies on every exit path so the spinner stops rendering.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                return {"run_id": run_id, "dataset_info": "data.jsonl"}

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )
        # Guard: the rewired handler must NOT shell out any more.
        def _no_subprocess(*args, **kwargs):
            raise AssertionError(
                "replay must NOT shell out; the in-process preflight uses "
                "RunHistoryManager."
            )

        monkeypatch.setattr("subprocess.run", _no_subprocess)

        state = RunDetailState()
        state.current_run_id = "rid-replay"

        state.replay()

        assert state.action_in_flight == "", (
            "FRONTEND-B-014-EXTENDED regression on replay: "
            "action_in_flight not cleared after the in-process replay "
            "preflight. The try/finally MUST clear the field on every "
            "exit path so the spinner stops rendering."
        )

    def test_action_in_flight_cleared_after_export_run(self, monkeypatch, tmp_path):
        """The same try/finally contract holds for export_run.

        UI-A-004 (Wave A1): export_run is in-process now (single-run JSONL
        to a sandboxed path); the try/finally contract still holds.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.current_run_id = "rid-export"

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                return {"run_id": "rid-export", "status": "completed"}

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

        state.export_run()

        assert state.action_in_flight == "", (
            "FRONTEND-B-014-EXTENDED regression on export_run: "
            "action_in_flight not cleared after the in-process export."
        )


# =============================================================================
# V2-a (Wave A2 verifier gap): the in-process action handlers
# (delete_run / export_run / replay), rewired in Wave A1 from broken
# subprocess shell-outs, write RAW absolute paths (the sandbox ``history_dir``
# / ``out_path``, both under ``~/.backpropagate/ui-outputs`` which embeds the
# OS home dir + username) AND the bare ``{exc}`` repr of an OSError into the
# PUBLIC (client-serialized) ``action_error`` / ``action_result`` vars. Under
# the documented ``--share + --auth`` flow those ship to the remote browser.
# Every path-bearing action string must be routed through ``_redact_paths``
# (via ``_redact_action``) before assignment.
# =============================================================================


class TestRunDetailActionStringsRedactPaths:
    """Path-bearing action_error / action_result strings must be redacted."""

    @staticmethod
    def _home_history_dir():
        """A synthetic history dir UNDER the user's home so the home prefix
        is present in any error string the handler builds. ``_redact_paths``
        targets ``/home/<user>/…`` / ``C:\\Users\\<user>\\…`` so a path under
        the real home is the deterministic, cross-platform probe.
        """
        from pathlib import Path as _Path

        return _Path.home() / ".backpropagate" / "ui-outputs"

    def _assert_no_home_leak(self, text: str, label: str) -> None:
        """Assert the operator's home dir / username does not appear raw."""
        from pathlib import Path as _Path

        home = str(_Path.home())
        assert home not in text, (
            f"V2-a: {label} leaks the operator home dir ({home!r}) into a "
            f"client-serialized var. Route the string through _redact_action "
            f"so the home prefix becomes <redacted-path>. Got: {text!r}"
        )
        # The username component specifically must not survive.
        try:
            user = _Path.home().name
        except Exception:  # noqa: BLE001
            user = ""
        if user:
            assert user not in text or "<redacted-path>" in text, (
                f"V2-a: {label} appears to leak the username ({user!r}); "
                f"expected a <redacted-path> token. Got: {text!r}"
            )

    def test_delete_run_not_found_error_redacts_history_dir(self, monkeypatch):
        """delete_run's not-found branch embeds ``history_dir`` (home+user) —
        it must be redacted before reaching ``action_error``.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        hist = self._home_history_dir()
        hist.mkdir(parents=True, exist_ok=True)

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                return False  # not found -> error string embeds history_dir

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir", lambda: hist
        )

        state = RunDetailState()
        state.current_run_id = "missingrun"
        state.delete_run()

        assert state.action_error, "expected a not-found error message"
        self._assert_no_home_leak(state.action_error, "delete_run not-found")
        assert "<redacted-path>" in state.action_error, (
            "V2-a: the redacted history_dir should surface as <redacted-path>."
        )

    def test_delete_run_exception_redacts_oserror_path(self, monkeypatch):
        """A raw OSError repr (``[Errno 2] … '/home/<user>/…'``) must be
        redacted on the exception branch of delete_run.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        hist = self._home_history_dir()
        hist.mkdir(parents=True, exist_ok=True)
        leaky = str(hist / "secret-run" / "manifest.json")

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                raise OSError(f"[Errno 13] Permission denied: '{leaky}'")

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir", lambda: hist
        )

        state = RunDetailState()
        state.current_run_id = "abc12345"
        state.delete_run()

        assert state.action_error, "expected an exception error message"
        self._assert_no_home_leak(state.action_error, "delete_run exception")

    def test_export_run_success_redacts_out_path(self, monkeypatch):
        """export_run's success string embeds the sandbox ``out_path``
        (home+user) — it must be redacted in ``action_result``.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        hist = self._home_history_dir()
        hist.mkdir(parents=True, exist_ok=True)

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                return {"run_id": run_id, "status": "completed"}

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir", lambda: hist
        )

        state = RunDetailState()
        state.current_run_id = "exportme"
        state.export_run()

        assert state.action_result, "expected a success message"
        self._assert_no_home_leak(state.action_result, "export_run success")
        assert "<redacted-path>" in state.action_result, (
            "V2-a: the sandbox out_path must surface as <redacted-path> in "
            "the client-serialized action_result."
        )

    def test_replay_not_found_error_redacts_history_dir(self, monkeypatch):
        """replay's not-found branch embeds ``history_dir`` — redact it."""
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        hist = self._home_history_dir()
        hist.mkdir(parents=True, exist_ok=True)

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                return None  # not found -> error embeds history_dir

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir", lambda: hist
        )

        state = RunDetailState()
        state.current_run_id = "ghostrun"
        state.replay()

        assert state.action_error, "expected a not-found error message"
        self._assert_no_home_leak(state.action_error, "replay not-found")


# =============================================================================
# UI-A-002 (Wave A2): absolute paths carrying the operator's home dir +
# username must NOT live in PUBLIC (client-serialized) Reflex vars. The full
# paths move to backend-only ('_'-prefixed) vars; the client sees only
# basenames / redacted forms via computed vars. ``log_lines`` is redacted
# per-line before assignment.
# =============================================================================


class TestUiPathFieldsNotClientSerialized:
    """Full-path fields must be backend-only; only redacted/basename forms
    are client-serialized.
    """

    def test_dataset_uploaded_path_is_backend_only(self):
        from backpropagate.ui_state import DatasetState

        assert "uploaded_path" not in DatasetState.base_vars, (
            "UI-A-002: DatasetState.uploaded_path (full path w/ home+user) "
            "must NOT be a client-serialized var."
        )
        assert "_uploaded_path" in DatasetState.backend_vars, (
            "UI-A-002: the full upload path must live in the backend-only "
            "var DatasetState._uploaded_path."
        )
        # The basename + boolean ARE the public surface.
        assert "uploaded_basename" in DatasetState.computed_vars
        assert "has_upload" in DatasetState.computed_vars

    def test_run_detail_checkpoint_path_is_backend_only(self):
        from backpropagate.ui_state import RunDetailState

        assert "checkpoint_path" not in RunDetailState.base_vars, (
            "UI-A-002: RunDetailState.checkpoint_path (full path) must NOT "
            "be a client-serialized var."
        )
        assert "_checkpoint_path" in RunDetailState.backend_vars, (
            "UI-A-002: the full checkpoint path must live in the backend-"
            "only var RunDetailState._checkpoint_path."
        )
        assert "checkpoint_path_display" in RunDetailState.computed_vars, (
            "UI-A-002: the redacted client-facing form must be a computed "
            "var (checkpoint_path_display)."
        )

    def test_models_cache_dir_is_backend_only(self):
        from backpropagate.ui_state import ModelsState

        assert "cache_dir" not in ModelsState.base_vars, (
            "UI-A-002: ModelsState.cache_dir (full ~/.cache path) must NOT "
            "be a client-serialized var."
        )
        assert "_cache_dir" in ModelsState.backend_vars, (
            "UI-A-002: the full cache dir must live in the backend-only var "
            "ModelsState._cache_dir."
        )
        assert "cache_dir_display" in ModelsState.computed_vars

    def test_checkpoint_path_display_redacts_home(self):
        from pathlib import Path as _Path

        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state._checkpoint_path = str(_Path.home() / "runs" / "run-x" / "ckpt")
        disp = state.checkpoint_path_display
        assert str(_Path.home()) not in disp, (
            f"UI-A-002: checkpoint_path_display leaks the home dir: {disp!r}"
        )
        assert "<redacted-path>" in disp

    def test_cache_dir_display_redacts_home(self):
        from pathlib import Path as _Path

        from backpropagate.ui_state import ModelsState

        state = ModelsState()
        state._cache_dir = str(_Path.home() / ".cache" / "huggingface" / "hub")
        disp = state.cache_dir_display
        assert str(_Path.home()) not in disp, (
            f"UI-A-002: cache_dir_display leaks the home dir: {disp!r}"
        )
        # ui-v2 P2: shown home-relative, not as a literal "<redacted-path>".
        assert disp == "~/.cache/huggingface/hub"

    def test_load_run_redacts_log_lines(self, monkeypatch, tmp_path):
        """A training.log line containing an absolute home path must be
        redacted before it enters the client-serialized ``log_lines`` var.
        """
        from pathlib import Path as _Path

        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        # Build a fake checkpoint dir with a training.log that embeds a home
        # path. Use the real tmp_path for the checkpoint dir, but write a log
        # line that contains the operator's actual home dir so the redaction
        # pattern fires deterministically.
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        leaky_line = f"loaded adapter from {_Path.home() / 'models' / 'a.safetensors'}"
        (ckpt / "training.log").write_text(leaky_line + "\n", encoding="utf-8")

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                return {
                    "run_id": run_id,
                    "status": "completed",
                    "checkpoint_path": str(ckpt),
                }

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir", lambda: tmp_path
        )

        state = RunDetailState()
        state.current_run_id = "withlogs"
        state.load_run()

        assert state.log_lines, "expected the training.log tail to populate"
        joined = "\n".join(state.log_lines)
        assert str(_Path.home()) not in joined, (
            f"UI-A-002: log_lines leaks the operator home dir: {joined!r}"
        )


# =============================================================================
# UI-A-005 (Wave A2): RunsState.load_runs reads run-history from
# ``output_dir_override`` — a PUBLIC var a malicious WS client can set
# directly, bypassing the validating setter. The forbidden-base guard must be
# re-applied at the READ sink, not only in the setter.
# =============================================================================


class TestRunsStateOutputDirOverrideReadGuard:
    """A forbidden output_dir_override must be refused at the read sink even
    when set directly (bypassing the setter).
    """

    def test_load_runs_refuses_forbidden_override_set_directly(self, monkeypatch):
        """Simulate a WS client writing the public var directly to a
        credential dir, then calling load_runs. The read sink must refuse.
        """
        from pathlib import Path as _Path

        from backpropagate.ui_state import RunsState

        state = RunsState()
        # Bypass set_output_dir_override entirely — assign the public var.
        state.output_dir_override = str(_Path.home() / ".ssh")

        # Guard: RunHistoryManager must NEVER be constructed for a forbidden
        # base (that would mean we read from it).
        from backpropagate import checkpoints as _ck

        def _boom(_dir):
            raise AssertionError(
                "UI-A-005: load_runs constructed RunHistoryManager on a "
                "forbidden base — the read-sink guard did not fire."
            )

        monkeypatch.setattr(_ck, "RunHistoryManager", _boom)

        state.load_runs()

        assert state.runs == []
        assert state.error, "expected a refusal error"
        assert "system or credential" in state.error.lower() or "refus" in state.error.lower(), (
            f"UI-A-005: expected a forbidden-base refusal; got {state.error!r}"
        )

    def test_load_runs_allows_benign_override(self, monkeypatch, tmp_path):
        """An override inside the UI output folder still loads.

        ``Path.home`` is pointed at the temp dir so the sandbox check does
        not create a directory under the real home.
        """
        from pathlib import Path as _Path

        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunsState

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)

        class _FakeManager:
            def __init__(self, _dir):
                self.directory = _dir

            def list_runs(self, status=None, limit=50):
                return []

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)

        inside = home / ".backpropagate" / "ui-outputs" / "custom"
        inside.mkdir(parents=True)
        state = RunsState()
        state.output_dir_override = str(inside)
        state.load_runs()

        assert state.runs == []
        assert state.error == ""


# =============================================================================
# UI-A-006 (Wave A2): delete_model must use is_relative_to confinement (not
# str.startswith) and refuse a symlinked cache entry before rmtree.
# =============================================================================


class TestModelsStateDeleteModelHardening:
    """delete_model confinement + symlink refusal."""

    def test_delete_model_refuses_symlink_entry(self, monkeypatch, tmp_path):
        """A ``models--*`` symlink in the cache must be refused (not followed
        into an rmtree of the link target).
        """
        from pathlib import Path as _Path

        from backpropagate.ui_state import ModelsState

        # Point the HF cache at a temp dir we control.
        fake_cache = tmp_path / ".cache" / "huggingface" / "hub"
        fake_cache.mkdir(parents=True)
        outside = tmp_path / "precious"
        outside.mkdir()
        (outside / "keep.txt").write_text("do not delete", encoding="utf-8")

        link = fake_cache / "models--evil--link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this host")

        # Redirect Path.home() so cache_dir resolves under tmp_path.
        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path))
        for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            monkeypatch.delenv(_var, raising=False)  # cache under the fake home

        state = ModelsState()
        state.delete_model("models--evil--link")

        assert state.error, "expected a refusal error for the symlink entry"
        assert "symlink" in state.error.lower(), (
            f"UI-A-006: expected a symlink refusal; got {state.error!r}"
        )
        # The link target must be untouched.
        assert (outside / "keep.txt").exists(), (
            "UI-A-006: delete_model followed a symlink and deleted the "
            "link target outside the cache."
        )
        # And the link itself must still exist (we refused, not deleted).
        assert link.is_symlink()

    def test_delete_model_sibling_prefix_dir_not_confused(self, monkeypatch, tmp_path):
        """A sibling dir whose path shares the cache prefix string must not be
        treated as inside the cache (is_relative_to vs startswith).

        This is exercised indirectly: a normal in-cache delete still works,
        proving the is_relative_to swap didn't break the happy path.
        """
        from pathlib import Path as _Path

        from backpropagate.ui_state import ModelsState

        fake_cache = tmp_path / ".cache" / "huggingface" / "hub"
        fake_cache.mkdir(parents=True)
        victim = fake_cache / "models--org--model"
        victim.mkdir()
        (victim / "blob").write_text("x", encoding="utf-8")

        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path))
        for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            monkeypatch.delenv(_var, raising=False)  # cache under the fake home
        # Avoid the post-delete reload doing real work.
        monkeypatch.setattr(ModelsState, "load_models", lambda self: None)

        state = ModelsState()
        state.delete_model("models--org--model")

        assert not victim.exists(), (
            "UI-A-006: a legitimate in-cache delete must still succeed after "
            "the is_relative_to swap."
        )
        assert state.error == "", f"unexpected error: {state.error!r}"


# =============================================================================
# CLIUI-B-005 (Stage C): delete_model double-click guard. Pre-fix the delete
# button had no in-flight disable, so a double-click re-entered delete_model;
# the second invocation found the directory already gone and surfaced a
# spurious "Model directory not found" error. The handler now carries a
# per-row in-flight flag (``deleting_dir``) that gates re-entry, mirroring
# RunDetailState.action_in_flight; the cleared-on-every-exit-path invariant
# is enforced via try/finally.
# =============================================================================


class TestModelsStateDeleteModelInFlight:
    """delete_model re-entrancy guard (CLIUI-B-005)."""

    def test_deleting_dir_defaults_empty(self):
        """``deleting_dir`` defaults to '' (no delete in flight on first render)."""
        from backpropagate.ui_state import ModelsState

        state = ModelsState()
        assert hasattr(state, "deleting_dir"), (
            "CLIUI-B-005 regression: ModelsState is missing the deleting_dir "
            "field. The per-row delete button has nothing to bind its "
            "in-flight disable to."
        )
        assert state.deleting_dir == ""

    def test_deleting_dir_cleared_after_successful_delete(self, monkeypatch, tmp_path):
        """After a successful delete the in-flight flag is cleared (try/finally)."""
        from pathlib import Path as _Path

        from backpropagate.ui_state import ModelsState

        fake_cache = tmp_path / ".cache" / "huggingface" / "hub"
        fake_cache.mkdir(parents=True)
        victim = fake_cache / "models--org--model"
        victim.mkdir()
        (victim / "blob").write_text("x", encoding="utf-8")

        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path))
        for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            monkeypatch.delenv(_var, raising=False)  # cache under the fake home
        monkeypatch.setattr(ModelsState, "load_models", lambda self: None)

        state = ModelsState()
        state.delete_model("models--org--model")

        assert not victim.exists()
        assert state.deleting_dir == "", (
            "CLIUI-B-005: deleting_dir must be cleared on the success path so "
            "the row's delete button re-enables."
        )

    def test_double_click_does_not_surface_spurious_not_found(self, monkeypatch, tmp_path):
        """A second delete of the SAME dir while one is in flight must be a
        no-op, not a 'not found' error.

        Pre-fix the second invocation found the directory already gone and
        latched ``error`` with a confusing "Model directory not found"
        message. The re-entrancy guard makes the second call a silent no-op.
        """
        from pathlib import Path as _Path

        from backpropagate.ui_state import ModelsState

        fake_cache = tmp_path / ".cache" / "huggingface" / "hub"
        fake_cache.mkdir(parents=True)
        victim = fake_cache / "models--org--model"
        victim.mkdir()
        (victim / "blob").write_text("x", encoding="utf-8")

        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path))
        for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            monkeypatch.delenv(_var, raising=False)  # cache under the fake home
        monkeypatch.setattr(ModelsState, "load_models", lambda self: None)

        state = ModelsState()
        # Simulate the in-flight window of a first click that hasn't finished
        # (the reload hasn't refreshed the row away yet).
        state.deleting_dir = "models--org--model"
        # The directory is already gone (first rmtree completed).
        import shutil as _shutil

        _shutil.rmtree(victim)

        state.delete_model("models--org--model")

        assert state.error == "", (
            "CLIUI-B-005: a re-entrant delete of the same dir while one is "
            f"in flight must be a no-op, not a spurious error; got "
            f"{state.error!r}."
        )

    def test_deleting_dir_cleared_when_delete_raises(self, monkeypatch, tmp_path):
        """If rmtree raises, the finally MUST still clear ``deleting_dir`` so
        the button doesn't latch disabled forever.
        """
        import shutil as _shutil
        from pathlib import Path as _Path

        from backpropagate.ui_state import ModelsState

        fake_cache = tmp_path / ".cache" / "huggingface" / "hub"
        fake_cache.mkdir(parents=True)
        victim = fake_cache / "models--org--model"
        victim.mkdir()

        monkeypatch.setattr(_Path, "home", classmethod(lambda cls: tmp_path))
        for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
            monkeypatch.delenv(_var, raising=False)  # cache under the fake home

        def _boom(*_a, **_k):
            raise OSError("disk on fire")

        monkeypatch.setattr(_shutil, "rmtree", _boom)

        state = ModelsState()
        state.delete_model("models--org--model")

        assert state.deleting_dir == "", (
            "CLIUI-B-005: deleting_dir must be cleared on the exception path "
            "(try/finally) so the delete button re-enables after a failure."
        )
        assert state.error, "a delete failure must still surface an operator error"


# =============================================================================
# UI-A-001 (Wave A1 CRITICAL): the HF token must NOT be a client-serialized
# (public) Reflex var. Pre-fix ``ExportState.hub_token`` was in ``base_vars``
# and bound two-way to a type=password input — the live write-scoped token
# round-tripped to the browser on every keystroke. The fix moves the raw
# secret to a backend-only (``_``-prefixed) var that Reflex never serializes
# to the client.
# =============================================================================


class TestExportStateHubTokenNotClientSerialized:
    """The raw HF token must live ONLY in a backend var, never in base_vars."""

    def test_hub_token_not_in_base_vars(self):
        """UI-A-001 CRITICAL regression: ``hub_token`` (the raw write-scoped
        credential) must NOT be a client-serialized Reflex var.

        Reflex serializes every name in ``base_vars`` into the WebSocket
        state bundle that ships to the browser. A secret in ``base_vars``
        round-trips to the client; the type=password mask is visual only.
        """
        from backpropagate.ui_state import ExportState

        assert "hub_token" not in ExportState.base_vars, (
            "UI-A-001 CRITICAL: ExportState.hub_token is a client-"
            "serialized (public) Reflex var. The raw HF token must be "
            "held in a backend-only ('_'-prefixed) var so it is never "
            "serialized into the WS state bundle sent to the browser."
        )

    def test_hub_token_is_not_a_state_var_of_any_kind(self):
        """The secret is in no state var: Reflex pickles backend vars to disk too."""
        from backpropagate.ui_state import ExportState

        assert "_hub_token" not in ExportState.backend_vars
        assert "_hub_token" not in ExportState.base_vars
        assert not [
            name
            for name in (*ExportState.backend_vars, *ExportState.base_vars)
            if "token" in name.lower() and "file" not in name and name not in (
                "hub_token_set", "hub_token_error"
            )
        ]

    def test_no_public_state_var_holds_a_secret(self):
        """Sibling probe: NO public (client-serialized) Reflex var across
        the UI state classes may hold a raw credential.

        Guards against re-introducing UI-A-001 on a sibling surface (a new
        token / password / secret field declared as a plain public var).
        Field *paths* (e.g. hub_token_file_path) and *error* sentinels are
        allow-listed because they are references / UI strings, not the
        secret material itself.
        """
        import reflex as rx

        from backpropagate import ui_state as _m

        secret_substrings = ("token", "password", "secret", "passwd", "apikey", "api_key")
        # Public vars whose name matches a secret keyword but which provably
        # hold NO credential material:
        #  - hub_token_file_path        -> a filesystem path reference, not the token
        #  - *_set (e.g. hub_token_set) -> boolean mirror carrying only "is a
        #    token captured", never the secret itself (the UI-A-001 design)
        #  - *tokens* (plural: min/max/avg_tokens) -> dataset token COUNTS (ints)
        allow = {"hub_token_file_path"}

        offenders: list[str] = []
        for name in dir(_m):
            obj = getattr(_m, name)
            if not isinstance(obj, type) or not issubclass(obj, rx.State):
                continue
            for var_name in obj.base_vars:
                low = var_name.lower()
                if (
                    var_name in allow
                    or low.endswith(("_error", "_path", "_set"))
                    or "tokens" in low  # plural -> dataset count, not a credential
                ):
                    continue
                if any(s in low for s in secret_substrings):
                    offenders.append(f"{obj.__name__}.{var_name}")

        assert not offenders, (
            "UI-A-001 sibling regression: public (client-serialized) "
            f"Reflex var(s) appear to hold a secret: {offenders}. Move "
            "the raw credential to a backend-only ('_'-prefixed) var."
        )

    def test_set_hub_token_writes_backend_var(self, monkeypatch):
        """The write-only setter populates the backend var, not a public one.

        ``set_hub_token`` writes the process-memory store. The
        public mirror that the form binds ``value=`` to (if any) must NOT
        contain the raw secret.
        """
        from backpropagate.ui_state import ExportState

        state = ExportState()
        state.set_hub_token("hf_" + "a" * 40)

        assert state._hub_token_value() == "hf_" + "a" * 40, (
            "UI-A-001: set_hub_token must keep the raw token for the push."
        )

    def test_push_to_hub_reads_and_clears_backend_token(self, monkeypatch, tmp_path):
        """``push_to_hub`` reads the backend token and clears it on success."""
        from pathlib import Path

        from backpropagate.ui_state import ExportState

        captured: dict[str, object] = {}

        def _fake_push(**kwargs):
            captured.update(kwargs)

        # ``push_to_hub`` does ``from .export import push_to_hub as _push``;
        # patch the source symbol the UI imports.
        import backpropagate.export as _export_mod

        monkeypatch.setattr(_export_mod, "push_to_hub", _fake_push, raising=False)

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
        adapter = home / ".backpropagate" / "ui-outputs" / "adapter"
        adapter.mkdir(parents=True)

        state = ExportState()
        state.set_source_model_path(str(adapter))
        state.set_hub_repo_id("owner/repo")
        state.set_hub_token("hf_" + "b" * 40)

        state.push_to_hub()

        assert captured.get("token") == "hf_" + "b" * 40, (
            "UI-A-001: push_to_hub must resolve the token from the "
            f"backend var and pass it to export.push_to_hub; got {captured!r}."
        )
        assert state._hub_token_value() == "", (
            "UI-A-001: push_to_hub must clear the backend token after a "
            "successful push so it doesn't sit in state for the WS session."
        )


# =============================================================================
# UI-A-003 (Wave A1 HIGH): user-controlled run IDs reach subprocess argv with
# no format validation. An option-shaped run_id (``--to=…`` / ``-x``) is
# parsed as a FLAG by the downstream CLI. The fix validates run IDs at the
# trust boundary (strict allowlist regex rejecting any leading ``-``) and
# inserts a ``--`` end-of-options separator before positional run IDs.
# =============================================================================


class TestRunDetailRunIdValidation:
    """Run-id format validation at the trust boundary (UI-A-003)."""

    def test_set_diff_other_run_id_rejects_option_shaped(self):
        """A leading-dash run id (``--to=evil``) is rejected, not stored."""
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.set_diff_other_run_id("--to=/etc/passwd")

        assert state.diff_other_run_id == "", (
            "UI-A-003 HIGH: an option-shaped comparison run id must be "
            "rejected at set_diff_other_run_id; storing it lets the "
            "downstream CLI parse it as a flag. Got "
            f"{state.diff_other_run_id!r}."
        )
        assert state.action_error != "", (
            "UI-A-003: rejecting a malformed run id must surface a clean "
            "operator-facing error."
        )

    def test_set_diff_other_run_id_rejects_bare_leading_dash(self):
        """Even a short ``-x`` (otherwise all-allowlist-chars) is rejected
        because a leading dash makes argparse treat it as a flag.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.set_diff_other_run_id("-x")

        assert state.diff_other_run_id == "", (
            "UI-A-003: a leading '-' must be rejected even when the rest "
            "of the value is in the allowlist char set ('-x' is a CLI "
            "flag to argparse)."
        )

    def test_set_diff_other_run_id_accepts_valid(self):
        """A well-formed run id passes through unchanged."""
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.set_diff_other_run_id("abc123_DEF-456")

        assert state.diff_other_run_id == "abc123_DEF-456"
        assert state.action_error == ""

    def test_set_diff_other_run_id_rejects_overlong(self):
        """A run id over 64 chars is rejected (DoS / argv-bloat guard)."""
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.set_diff_other_run_id("a" * 65)

        assert state.diff_other_run_id == "", (
            "UI-A-003: run ids longer than 64 chars must be rejected."
        )

    def test_load_run_rejects_option_shaped_route_param(self, monkeypatch, tmp_path):
        """A malicious ``rid`` route param (``--to=…``) is rejected before
        it can reach current_run_id / any shell-out.
        """
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()

        class _FakeParams:
            @staticmethod
            def get(key, default=None):
                return "--to=/etc/passwd" if key == "rid" else default

        class _FakePage:
            params = _FakeParams()

        class _FakeRouter:
            page = _FakePage()

        monkeypatch.setattr(type(state), "router", _FakeRouter(), raising=False)

        state.load_run()

        assert state.current_run_id != "--to=/etc/passwd", (
            "UI-A-003: an option-shaped rid route param must NOT be "
            "assigned to current_run_id (it would flow into subprocess "
            f"argv). Got {state.current_run_id!r}."
        )
        assert state.error != "", (
            "UI-A-003: a rejected route param must surface a clean error."
        )

    def test_diff_against_inserts_end_of_options_separator(self, monkeypatch):
        """Defense-in-depth: the diff-runs shell-out inserts ``--`` before
        the positional run IDs so even a (hypothetically) dash-leading id
        is treated as a positional, not a flag.
        """
        from backpropagate.ui_state import RunDetailState

        captured_argv: dict[str, list] = {}

        def _fake_run(argv, *args, **kwargs):
            captured_argv["argv"] = argv
            from unittest.mock import MagicMock

            r = MagicMock()
            r.returncode = 0
            r.stdout = "ok"
            r.stderr = ""
            return r

        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/local/bin/backprop"
            if name in ("backprop", "backpropagate")
            else None,
        )
        monkeypatch.setattr("subprocess.run", _fake_run)

        state = RunDetailState()
        state.current_run_id = "run-a"
        state.diff_against("run-b")

        argv = captured_argv["argv"]
        assert "--" in argv, (
            "UI-A-003: diff_against must insert a '--' end-of-options "
            f"separator before the positional run IDs. argv={argv!r}."
        )
        sep = argv.index("--")
        assert argv[sep + 1 :] == ["run-a", "run-b"], (
            "UI-A-003: both run IDs must follow the '--' separator as "
            f"positionals. argv={argv!r}."
        )


# =============================================================================
# UI-A-004 (Wave A1 MED, coupled to A-003): delete_run / export_run shelled
# out to a NON-EXISTENT subcommand / flag (``delete-run``; ``export-runs
# --run-id``) → argparse error → broken buttons. The fix rewires both to the
# EXISTING surface: delete uses RunHistoryManager in-process; export writes a
# single-run JSONL to a SANDBOXED path. These smoke tests assert the real
# argparse accepts the shapes we DO shell out to (diff-runs / replay) and
# that the rewired handlers no longer depend on the phantom surface.
# =============================================================================


class TestRunDetailShellOutsParseAgainstRealArgparse:
    """The CLI argv shapes the UI emits must parse against create_parser()."""

    def test_diff_runs_argv_parses(self, cli_parser):
        """``diff-runs -- <a> <b>`` (the UI-A-003 shape) parses cleanly."""
        args = cli_parser.parse_args(["diff-runs", "--", "run-a", "run-b"])
        assert args.command == "diff-runs"
        assert args.run_id_a == "run-a"
        assert args.run_id_b == "run-b"

    def test_replay_positional_after_separator_parses(self, cli_parser):
        """The ``--``-then-positional shape (UI-A-003 separator) parses.

        The ``replay`` handler no longer shells out at all — it was rewired
        to an in-process preflight (see TestRunDetailReplayInProcess) once
        the phantom ``--dry-run`` token the old shell-out passed was confirmed
        to be rejected by argparse (test_replay_has_no_dry_run_flag pins that
        root cause). This stays as a CLI-surface pin: ``replay -- <id>``
        treats the id as a positional, never a flag.
        """
        args = cli_parser.parse_args(["replay", "--", "run-a"])
        assert args.command == "replay"
        assert args.run_id == "run-a"

    def test_replay_has_no_dry_run_flag(self, cli_parser):
        """Pin the root cause: ``replay --dry-run`` is NOT a real flag.

        This is why the pre-fix UI shell-out (``replay --dry-run -- <id>``)
        failed at argparse (``unrecognized arguments: --dry-run``). The fix
        rewires replay to an in-process preflight; this test documents the
        phantom flag so a future edit doesn't re-introduce the shell-out.
        Sibling to test_export_runs_has_no_run_id_flag /
        test_no_delete_run_subcommand.

        NOTE: this asserts the CURRENT cli.py surface (replay has no
        ``--dry-run``). A real ``--dry-run`` flag on the replay subcommand is
        a separately-tracked deferred feature owned by the CLI domain; if it
        lands, this pin flips to assert the flag parses instead.
        """
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["replay", "--dry-run", "--", "abc"])

    def test_export_runs_has_no_run_id_flag(self, cli_parser):
        """Pin the root cause: ``export-runs --run-id`` is NOT a real flag.

        This is why the pre-fix UI shell-out (``export-runs --run-id <id>``)
        failed at argparse. The fix rewires export_run to in-process /
        valid-flag export; this test documents the phantom flag so a future
        edit doesn't re-introduce it.
        """
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["export-runs", "--run-id", "abc"])

    def test_no_delete_run_subcommand(self, cli_parser):
        """Pin the root cause: there is NO ``delete-run`` subcommand.

        The pre-fix UI shelled out to ``backprop delete-run <id> --yes``,
        which argparse rejected. delete_run is now in-process; this test
        documents the phantom subcommand.
        """
        with pytest.raises(SystemExit):
            cli_parser.parse_args(["delete-run", "abc", "--yes"])


class TestRunDetailDeleteRunInProcess:
    """delete_run uses RunHistoryManager directly (UI-A-004) — no subprocess."""

    def test_delete_run_calls_run_history_manager(self, monkeypatch, tmp_path):
        """A successful in-process delete sets was_deleted + action_result."""
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        calls: dict[str, str] = {}

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                calls["run_id"] = run_id
                return True

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )
        # Guard: if the handler still shells out, fail loudly.
        def _no_subprocess(*args, **kwargs):
            raise AssertionError(
                "UI-A-004: delete_run must NOT shell out; use "
                "RunHistoryManager in-process."
            )

        monkeypatch.setattr("subprocess.run", _no_subprocess)

        state = RunDetailState()
        state.current_run_id = "deadbeef"
        state.delete_run()

        assert calls.get("run_id") == "deadbeef", (
            "UI-A-004: delete_run must call RunHistoryManager.delete_run "
            f"in-process with the current run id; got {calls!r}."
        )
        assert state.was_deleted is True
        assert "deleted" in state.action_result.lower()
        assert state.action_error == ""

    def test_delete_run_missing_entry_surfaces_error(self, monkeypatch, tmp_path):
        """delete_run returning False (entry not found) surfaces an error
        and does NOT set was_deleted.
        """
        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def delete_run(self, run_id):
                return False

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

        state = RunDetailState()
        state.current_run_id = "ghost"
        state.delete_run()

        assert state.was_deleted is False
        assert state.action_error != ""


class TestRunDetailExportRunInProcess:
    """export_run writes a sandboxed single-run JSONL (UI-A-004)."""

    def test_export_run_writes_sandboxed_jsonl(self, monkeypatch, tmp_path):
        """export_run must produce a JSONL file under the sandbox dir and
        NOT shell out to the phantom ``export-runs --run-id`` flag.
        """
        import json

        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunDetailState

        run_entry = {"run_id": "rid-export", "status": "completed", "final_loss": 0.5}

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                return run_entry if run_id == "rid-export" else None

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

        def _no_subprocess(*args, **kwargs):
            raise AssertionError(
                "UI-A-004: export_run must NOT shell out to the phantom "
                "`export-runs --run-id` flag."
            )

        monkeypatch.setattr("subprocess.run", _no_subprocess)

        state = RunDetailState()
        state.current_run_id = "rid-export"
        state.export_run()

        assert state.action_error == "", (
            f"export_run should succeed in-process; got error "
            f"{state.action_error!r}."
        )
        # A JSONL file was written somewhere under the sandbox dir.
        written = list(tmp_path.rglob("*.jsonl"))
        assert written, (
            "UI-A-004: export_run must write a JSONL file under the "
            f"sandbox dir {tmp_path}; found none."
        )
        # The written file resolves INSIDE the sandbox (no traversal).
        for p in written:
            assert tmp_path.resolve() in p.resolve().parents or p.resolve().parent == tmp_path.resolve(), (
                f"UI-A-004: export path {p} escaped the sandbox {tmp_path}."
            )
        # And it contains the run record.
        record = json.loads(written[0].read_text(encoding="utf-8").splitlines()[0])
        assert record["run_id"] == "rid-export"


class TestRunDetailReplayInProcess:
    """replay() is an in-process preflight via RunHistoryManager — no
    subprocess, no phantom ``--dry-run`` flag.

    Sibling to the UI-A-004 delete_run / export_run rewiring: the Replay
    button validates that the run *can* be replayed (it exists and carries
    the one hard precondition cmd_replay enforces — a recorded
    ``dataset_info``) and then directs the operator to the real shell
    command, all without shelling out to a flag that does not exist.
    """

    @staticmethod
    def _patch(monkeypatch, tmp_path, entry):
        """Patch ``RunHistoryManager.get_run`` to return ``entry`` (matched
        by run_id; ``None`` otherwise), sandbox the history dir at
        ``tmp_path``, and hard-fail ANY shell-out so every test in this class
        implicitly proves the handler stays in-process.
        """
        from backpropagate import checkpoints as _ck

        class _FakeManager:
            def __init__(self, _dir):
                pass

            def get_run(self, run_id):
                if entry is not None and run_id == entry.get("run_id"):
                    return entry
                return None

        monkeypatch.setattr(_ck, "RunHistoryManager", _FakeManager)
        monkeypatch.setattr(
            "backpropagate.ui_security.get_ui_output_dir",
            lambda: tmp_path,
        )

        def _no_subprocess(*args, **kwargs):
            raise AssertionError(
                "replay must NOT shell out: the in-process preflight uses "
                "RunHistoryManager (the phantom `replay --dry-run` flag the "
                "old shell-out passed does not exist in cli.py)."
            )

        monkeypatch.setattr("subprocess.run", _no_subprocess)

    def test_replay_ok_when_run_is_replayable(self, monkeypatch, tmp_path):
        """A run with a recorded dataset_info preflights OK and hands the
        operator the real shell command for the heavy replay (it must NOT
        claim the replay already ran).
        """
        from backpropagate.ui_state import RunDetailState

        entry = {
            "run_id": "rid-replay",
            "dataset_info": "data.jsonl",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "session_kind": "single_run",
        }
        self._patch(monkeypatch, tmp_path, entry)

        state = RunDetailState()
        state.current_run_id = "rid-replay"
        state.replay()

        assert state.action_error == "", (
            f"replay preflight should succeed for a replayable run; got "
            f"error {state.action_error!r}."
        )
        assert "replayable" in state.action_result.lower()
        assert "backprop replay rid-replay" in state.action_result, (
            "The preflight result must hand the operator the real shell "
            f"command, not claim the replay ran; got {state.action_result!r}."
        )

    def test_replay_rejects_run_without_dataset_info(self, monkeypatch, tmp_path):
        """Mirror cmd_replay: a run with no dataset_info is NOT replayable —
        the preflight must surface the same verdict, not a false "OK".
        """
        from backpropagate.ui_state import RunDetailState

        entry = {"run_id": "rid-nodata", "dataset_info": None, "model_name": "m"}
        self._patch(monkeypatch, tmp_path, entry)

        state = RunDetailState()
        state.current_run_id = "rid-nodata"
        state.replay()

        assert state.action_result == ""
        assert "dataset_info" in state.action_error, (
            "A run with no dataset_info must surface the same 'cannot "
            "replay automatically' verdict cmd_replay returns "
            f"EXIT_USER_ERROR for; got {state.action_error!r}."
        )

    def test_replay_missing_entry_surfaces_error(self, monkeypatch, tmp_path):
        """get_run returning None (unknown run id) surfaces a not-found
        error and no result.
        """
        from backpropagate.ui_state import RunDetailState

        self._patch(
            monkeypatch, tmp_path,
            entry={"run_id": "other", "dataset_info": "d"},
        )

        state = RunDetailState()
        state.current_run_id = "ghost"
        state.replay()

        assert state.action_result == ""
        assert "not found" in state.action_error.lower()

    def test_replay_does_not_shell_out(self, monkeypatch, tmp_path):
        """The named regression guard for the original bug: replay must NEVER
        invoke ``subprocess.run`` (pre-fix it shelled out to the phantom
        ``replay --dry-run`` flag and always failed at argparse). ``_patch``
        installs a ``subprocess.run`` that raises if called; a clean run
        proves the handler stays fully in-process.
        """
        from backpropagate.ui_state import RunDetailState

        entry = {"run_id": "rid-noshell", "dataset_info": "data.jsonl"}
        self._patch(monkeypatch, tmp_path, entry)

        state = RunDetailState()
        state.current_run_id = "rid-noshell"
        state.replay()  # raises AssertionError inside the handler if it shells out

        assert state.action_error == ""
        assert state.action_in_flight == ""

    def test_replay_no_run_loaded(self, monkeypatch, tmp_path):
        """No current_run_id → clean error, no manager call, no shell-out."""
        from backpropagate.ui_state import RunDetailState

        self._patch(monkeypatch, tmp_path, entry={"run_id": "x", "dataset_info": "d"})

        state = RunDetailState()
        state.current_run_id = ""
        state.replay()

        assert state.action_error == "No run loaded."

    def test_replay_rejects_malformed_run_id(self, monkeypatch, tmp_path):
        """An option-shaped / malformed run id is rejected at the boundary
        (UI-A-003 _validate_run_id) before any lookup — preserved through the
        in-process rewire.
        """
        from backpropagate.ui_state import RunDetailState

        self._patch(monkeypatch, tmp_path, entry={"run_id": "x", "dataset_info": "d"})

        state = RunDetailState()
        state.current_run_id = "--malicious"
        state.replay()

        assert state.action_result == ""
        assert "invalid run id" in state.action_error.lower()


# =============================================================================
# HUX-02 (Stage C humanization): the operator-actionable remedy from
# ``sanitize_error_for_user`` must be held SEPARATELY from ``error`` (on the
# dedicated ``error_suggestion`` var) so the error callout can render it on its
# own dimmed ``hint=`` line instead of folding it into the message as a run-on
# "… Try: {suggestion}" sentence.
# =============================================================================


class TestRunsStateErrorSuggestionSplit:
    """``RunsState.load_runs`` exposes the remedy on ``error_suggestion``, kept
    out of the ``error`` message (HUX-02).
    """

    def test_suggestion_available_separately_not_run_on(self, monkeypatch, tmp_path):
        """A BackpropagateError with a suggestion routes its remedy to the
        dedicated ``error_suggestion`` var; ``error`` carries only the message
        (no "Try:" run-on tail).
        """
        from pathlib import Path

        from backpropagate import checkpoints as _ck
        from backpropagate.exceptions import BackpropagateError
        from backpropagate.ui_state import RunsState

        class _RaisingManager:
            def __init__(self, _dir):
                pass

            def list_runs(self, status=None, limit=50):
                raise BackpropagateError(
                    "history index is corrupt",
                    code="STATE_RUN_HISTORY_CORRUPT",
                    suggestion="delete the index and reload",
                )

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
        inside = home / ".backpropagate" / "ui-outputs"
        inside.mkdir(parents=True)
        monkeypatch.setattr(_ck, "RunHistoryManager", _RaisingManager)

        state = RunsState()
        state.output_dir_override = str(inside)
        state.load_runs()

        # The remedy lives on its own var, scannable as a distinct hint line.
        assert state.error_suggestion == "delete the index and reload"
        # The message carries the error but NOT the folded-in remedy.
        assert "history index is corrupt" in state.error
        assert "Try:" not in state.error
        assert "delete the index and reload" not in state.error
        assert state.runs == []

    def test_no_suggestion_leaves_hint_empty(self, monkeypatch, tmp_path):
        """A non-BackpropagateError (generic, suggestion=None) leaves
        ``error_suggestion`` empty so the callout hint collapses to a fragment
        (backward-compatible).
        """
        from pathlib import Path

        from backpropagate import checkpoints as _ck
        from backpropagate.ui_state import RunsState

        class _RaisingManager:
            def __init__(self, _dir):
                pass

            def list_runs(self, status=None, limit=50):
                raise RuntimeError("opaque internal failure")

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
        inside = home / ".backpropagate" / "ui-outputs"
        inside.mkdir(parents=True)
        monkeypatch.setattr(_ck, "RunHistoryManager", _RaisingManager)

        state = RunsState()
        state.output_dir_override = str(inside)
        state.load_runs()

        assert state.error_suggestion == ""
        assert state.error  # a generic, sanitized message is still surfaced
        assert "Try:" not in state.error

    def test_clear_error_resets_suggestion(self):
        """``clear_error`` wipes both ``error`` and ``error_suggestion`` so a
        dismissed banner leaves no stale hint behind.
        """
        from backpropagate.ui_state import RunsState

        state = RunsState()
        state.error = "something broke"
        state.error_suggestion = "do the thing"
        state.clear_error()

        assert state.error == ""
        assert state.error_suggestion == ""


class TestErrorSuggestionSlotsBackwardCompatible:
    """The /models and /run-detail callouts have a dedicated suggestion var so
    a future suggestion-bearing error surfaces in the same scannable hierarchy;
    both default empty today (HUX-02 backward compatibility).
    """

    def test_models_state_error_suggestion_defaults_empty(self):
        from backpropagate.ui_state import ModelsState

        assert ModelsState().error_suggestion == ""

    def test_run_detail_action_error_suggestion_defaults_empty(self):
        from backpropagate.ui_state import RunDetailState

        assert RunDetailState().action_error_suggestion == ""

    def test_clear_action_message_resets_suggestion(self):
        from backpropagate.ui_state import RunDetailState

        state = RunDetailState()
        state.action_error = "action broke"
        state.action_error_suggestion = "retry it"
        state.clear_action_message()

        assert state.action_error == ""
        assert state.action_error_suggestion == ""


# =============================================================================
# TrainState — ui-v2 P1 fix round (attach, heartbeat/ETA, stopped state)
# =============================================================================


class TestTrainStateAttach:
    """attach_active_job: finalize-fast on same-session terminal, adopt a
    live foreign run, and never double-start the poller."""

    def test_attach_finalizes_a_terminal_same_session_job(self, monkeypatch):
        """Lead fix #5: reload right after the end must not show 'active'."""
        import backpropagate.ui_jobs as ui_jobs
        from backpropagate.ui_state import TrainState

        class _Mgr:
            def status(self):
                return {"status": "stopped", "output_path": "out/x", "job_id": "run_x"}

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _Mgr())
        state = TrainState()
        state.job_id = "run_x"
        state.run_state = "active"
        result = state.attach_active_job()
        assert state.run_state == "stopped"
        assert state.job_output_path == "out/x"
        assert result is None  # no poller started for a dead job

    def test_attach_adopts_a_live_foreign_run(self, monkeypatch):
        import backpropagate.ui_jobs as ui_jobs
        from backpropagate.ui_state import TrainState

        class _Mgr:
            def status(self):
                return {
                    "status": "active",
                    "job_id": "run_foreign",
                    "phase": "training",
                    "total_steps": 400,
                    "step": 26,
                }

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _Mgr())
        state = TrainState()
        result = state.attach_active_job()
        assert state.job_id == "run_foreign"
        assert state.run_state == "active"
        assert state.job_total_steps == 400
        assert state.current_step == 26
        # The reattach banner is the ONLY recovery-class banner (fix #4).
        assert "Reattached" in state.reattach_notice
        assert result is TrainState.poll_job

    def test_attach_noop_when_nothing_running(self, monkeypatch):
        import backpropagate.ui_jobs as ui_jobs
        from backpropagate.ui_state import TrainState

        class _Mgr:
            def status(self):
                return {"status": "idle"}

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _Mgr())
        state = TrainState()
        assert state.attach_active_job() is None
        assert state.job_id == "" and state.reattach_notice == ""

    def test_attach_does_not_double_start_the_poller(self, monkeypatch):
        import backpropagate.ui_jobs as ui_jobs
        from backpropagate.ui_state import TrainState

        class _Mgr:
            def status(self):
                return {"status": "active", "job_id": "run_x", "phase": "training"}

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _Mgr())
        state = TrainState()
        state.job_id = "run_x"
        state.run_state = "active"
        state._poll_running = True  # a poller is already tailing
        assert state.attach_active_job() is None


class TestTrainStateLiveLabels:
    """Requirement 10: heartbeat 'last step N s ago' + ETA range."""

    def test_heartbeat_counts_seconds_since_last_step(self):
        import time as _time

        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.run_state = "active"
        state.job_phase = "training"
        state._last_step_epoch = _time.time() - 3
        state._tick_live_labels()
        assert state.heartbeat_label.startswith("last step ")
        assert "s ago" in state.heartbeat_label

    def test_heartbeat_silent_outside_training_phase(self):
        import time as _time

        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.run_state = "active"
        state.job_phase = "loading"
        state._last_step_epoch = _time.time() - 5
        state._tick_live_labels()
        assert state.heartbeat_label == ""

    def test_eta_range_after_warmup(self):
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.run_state = "active"
        state.job_phase = "training"
        state.job_total_steps = 100
        state.current_step = 50
        state._step_samples = 6
        state._step_time_ms_ema = 1000.0  # 1 s/step → ~50 s left
        import time as _time

        state._last_step_epoch = _time.time()
        state._tick_live_labels()
        assert state.eta_label == "about 40 s–65 s left"

    def test_eta_blank_before_warmup(self):
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.run_state = "active"
        state.job_phase = "training"
        state.job_total_steps = 100
        state.current_step = 2
        state._step_samples = 2
        state._step_time_ms_ema = 1000.0
        state._tick_live_labels()
        assert state.eta_label == ""

    def test_eta_shows_after_twenty_steps_with_ten_step_logging(self):
        """Steps are logged every 10, so step 30 of 420 has 3 samples. The
        warm-up is ~20 steps or 5%, so the ETA must show by now (a 5-sample
        floor hid it until step 50)."""
        from backpropagate.ui_state import TrainState

        state = TrainState()
        state.run_state = "active"
        state.job_phase = "training"
        state.job_total_steps = 420
        state.current_step = 30
        state._step_samples = 3
        state._step_time_ms_ema = 450.0
        import time as _time

        state._last_step_epoch = _time.time()
        state._tick_live_labels()
        assert state.eta_label.startswith("about ")


def test_merge_job_history_rows_merges_jobs_tree(tmp_path):
    """Fix #6: UI runs live under jobs/<run>/output — merged into Runs."""
    from backpropagate.ui_state import _merge_job_history_rows

    root_row = {"run_id": "cli_run", "started_at": "2026-10-01T10:00:00Z",
                "status": "completed"}
    job_out = tmp_path / "jobs" / "run_ui1" / "output"
    job_out.mkdir(parents=True)
    (job_out / "run_history.json").write_text(
        '[{"run_id": "ui_run", "started_at": "2026-10-01T12:00:00Z", '
        '"status": "completed"}]'
    )
    merged = _merge_job_history_rows([root_row], tmp_path, status=None, limit=50)
    ids = [r["run_id"] for r in merged]
    assert ids == ["ui_run", "cli_run"]  # started_at desc: job row first
    # The status filter applies to the job rows (root rows arrive pre-filtered
    # from RunHistoryManager.list_runs).
    filtered = _merge_job_history_rows([], tmp_path, status="failed", limit=50)
    assert filtered == []
    filtered = _merge_job_history_rows([], tmp_path, status="completed", limit=50)
    assert [r["run_id"] for r in filtered] == ["ui_run"]
