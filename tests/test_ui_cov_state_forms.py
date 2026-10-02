"""Behaviour tests for the form-side Reflex state classes and their helpers.

Covers ``AppState`` / ``TrainState`` / ``MultiRunState`` / ``ExportState`` /
``AuthBadgeState`` plus the module-level validators in ``ui_state`` (path
sandbox, run-id allowlist, token-file path, redaction, numeric coercion).

State classes are driven by calling the event handlers directly on an
instance (the established pattern - Reflex allows instantiation under pytest)
and asserting the field changes, companion ``*_error`` strings and side
effects.

Mocked, and why: ``backpropagate.export.push_to_hub`` (network: the HF Hub) is
replaced with a recorder so the tests assert exactly what the UI passed to it;
nothing else is mocked except ``Path.home`` / env vars, which point the
sandbox at ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("reflex", reason="reflex is required (install backpropagate[ui])")

from backpropagate import ui_state as us  # noqa: E402
from backpropagate.exceptions import BackpropagateError  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Isolated fake HOME + default UI output dir under ``tmp_path``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("BACKPROPAGATE_UI__OUTPUT_DIR", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    return SimpleNamespace(
        home=home.resolve(),
        out=(home / ".backpropagate" / "ui-outputs").resolve(),
    )


# =============================================================================
# Path sandbox helper
# =============================================================================


class TestValidateUiPath:
    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_blank_is_a_clearing_passthrough(self, value):
        assert us._validate_ui_path(value) == ("", "")

    def test_absolute_path_inside_sandbox_is_accepted_resolved(self, sandbox):
        target = sandbox.out / "runs" / "run-1" / "adapter"
        cleaned, err = us._validate_ui_path(f"  {target}  ")
        assert err == "" and Path(cleaned) == target

    @pytest.mark.parametrize(
        "evil",
        ["/etc/passwd", "C:\\Windows\\System32", "../../../etc/shadow", "..", "~/.ssh/id_rsa"],
    )
    def test_paths_outside_the_sandbox_are_refused(self, sandbox, evil, monkeypatch):
        monkeypatch.chdir(sandbox.home)
        cleaned, err = us._validate_ui_path(evil)
        assert cleaned == "" and err.startswith("Invalid path:")

    def test_dotdot_that_escapes_after_a_valid_prefix_is_refused(self, sandbox):
        sneaky = f"{sandbox.out}/runs/../../../../outside"
        cleaned, err = us._validate_ui_path(sneaky)
        assert cleaned == "" and err.startswith("Invalid path:")

    def test_relative_path_resolves_against_cwd_not_the_sandbox(self, sandbox, monkeypatch):
        """Documented behaviour: a relative path is only accepted when the cwd puts it inside."""
        sandbox.out.mkdir(parents=True)
        monkeypatch.chdir(sandbox.out)
        cleaned, err = us._validate_ui_path("runs/run-x/adapter")
        assert err == "" and Path(cleaned) == sandbox.out / "runs" / "run-x" / "adapter"
        monkeypatch.chdir(sandbox.home)
        assert us._validate_ui_path("runs/run-x/adapter")[1].startswith("Invalid path:")

    def test_nul_byte_is_refused(self, sandbox):
        cleaned, err = us._validate_ui_path(f"{sandbox.out}/a\x00b")
        assert cleaned == "" and err.startswith("Invalid path:")

    def test_forbidden_output_dir_override_makes_every_path_invalid(self, sandbox, monkeypatch):
        monkeypatch.setenv("BACKPROPAGATE_UI__OUTPUT_DIR", str(sandbox.home / ".ssh"))
        cleaned, err = us._validate_ui_path(str(sandbox.home / ".ssh" / "key"))
        assert cleaned == "" and "UI_OUTPUT_DIR_FORBIDDEN" in err or "forbidden" in err


class TestSetterPathFieldsUseTheSandbox:
    """Every state class path setter routes through the same sandbox helper."""

    def test_train_dataset_path(self, sandbox):
        s = us.TrainState()
        s.set_dataset_path(str(sandbox.out / "d.jsonl"))
        assert s.dataset_path == str(sandbox.out / "d.jsonl") and s.dataset_path_error == ""
        s.set_dataset_path("/etc/passwd")
        assert s.dataset_path == "" and s.dataset_path_error.startswith("Invalid path:")
        s.set_dataset_path("")
        assert (s.dataset_path, s.dataset_path_error) == ("", "")

    def test_multi_run_dataset_path(self, sandbox):
        s = us.MultiRunState()
        s.set_dataset_path("/etc/passwd")
        assert s.dataset_path == "" and s.dataset_path_error.startswith("Invalid path:")
        s.set_dataset_path(str(sandbox.out / "x.json"))
        assert s.dataset_path_error == "" and s.dataset_path.endswith("x.json")

    def test_export_source_model_path(self, sandbox):
        s = us.ExportState()
        s.set_source_model_path("/root/.ssh")
        assert s.source_model_path == "" and s.source_model_path_error
        s.set_source_model_path(str(sandbox.out / "adapter"))
        assert s.source_model_path_error == "" and s.source_model_path.endswith("adapter")

    @pytest.mark.parametrize("cls", [us.TrainState, us.MultiRunState])
    def test_model_field_only_sandboxes_absolute_paths(self, sandbox, cls):
        s = cls()
        s.set_model("meta-llama/Llama-3.1-8B")  # HF repo id: left alone
        assert (s.model, s.model_error) == ("meta-llama/Llama-3.1-8B", "")
        outside = str(sandbox.home.parent / "elsewhere" / "model")  # absolute on every OS
        s.set_model(outside)  # absolute local path: sandboxed
        assert s.model == "" and s.model_error.startswith("Invalid path:")
        inside = str(sandbox.out / "merged")
        s.set_model(inside)
        assert s.model == inside and s.model_error == ""
        s.set_model("")
        assert (s.model, s.model_error) == ("", "")


# =============================================================================
# Run-id allowlist, token file validation, redaction
# =============================================================================


class TestValidateRunId:
    @pytest.mark.parametrize("ok", ["abc123", "A_b-c", "0", "_x", "a" * 64, "  padded  "])
    def test_accepts_allowlisted_ids(self, ok):
        cleaned, err = us._validate_run_id(ok)
        assert err == "" and cleaned == ok.strip()

    @pytest.mark.parametrize(
        "bad",
        ["--to=/etc/passwd", "-o", "-x", "a" * 65, "a/b", "a\\b", "../x", "a b", "a;rm -rf",
         "a$(id)", "a`id`", "a\nb", "run.1", "é", "a\x00b", "%2e%2e"],
    )
    def test_rejects_flag_shaped_traversal_and_metacharacter_ids(self, bad):
        cleaned, err = us._validate_run_id(bad)
        assert cleaned == "" and err.startswith("Invalid run id")

    @pytest.mark.parametrize("blank", ["", "  ", "\n"])
    def test_blank_passthrough(self, blank):
        assert us._validate_run_id(blank) == ("", "")


class TestValidateTokenFilePath:
    def test_blank_passthrough(self):
        assert us._validate_token_file_path("  ") == ("", "")

    def test_nul_byte_refused_before_touching_disk(self):
        cleaned, err = us._validate_token_file_path("tok\x00en")
        assert cleaned == "" and "NUL byte" in err

    def test_existing_regular_file_is_resolved(self, tmp_path):
        f = tmp_path / "hf-token"
        f.write_text("hf_x" * 10, encoding="utf-8")
        cleaned, err = us._validate_token_file_path(f"  {f}  ")
        assert err == "" and Path(cleaned) == f.resolve()

    def test_missing_file_names_the_fix(self, tmp_path):
        cleaned, err = us._validate_token_file_path(str(tmp_path / "nope"))
        assert cleaned == "" and "does not exist" in err and "printf" in err

    def test_directory_is_refused(self, tmp_path):
        cleaned, err = us._validate_token_file_path(str(tmp_path))
        assert cleaned == "" and "not a regular file" in err

    def test_resolve_failure_falls_back_to_unresolved_path(self, tmp_path, monkeypatch):
        """Mocked: ``Path.resolve`` raises ``RuntimeError`` (symlink loop)."""
        f = tmp_path / "tok"
        f.write_text("x", encoding="utf-8")
        real = Path.resolve

        def boom(self, *a, **k):
            if self.name == "tok":
                raise RuntimeError("Symlink loop")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "resolve", boom)
        cleaned, err = us._validate_token_file_path(str(f))
        assert err == "" and cleaned == str(f)

    def test_stat_failure_is_reported_as_invalid(self, tmp_path, monkeypatch):
        """Mocked: ``Path.exists`` raises ``OSError`` (permission / stale handle)."""
        monkeypatch.setattr(Path, "exists", lambda self: (_ for _ in ()).throw(OSError("denied")))
        cleaned, err = us._validate_token_file_path(str(tmp_path / "x"))
        assert cleaned == "" and err.startswith("Invalid token-file path:")


class TestRedactAction:
    @pytest.mark.parametrize(
        ("raw", "gone"),
        [("/home/alice/.backpropagate/ui-outputs", "alice"),
         ("C:\\Users\\bob\\.cache\\hf", "bob"),
         ("/Users/carol/x", "carol")],
    )
    def test_home_prefixes_are_replaced(self, raw, gone):
        out = us._redact_action(f"No run history at {raw}.")
        assert gone not in out and "<redacted-path>" in out and out.startswith("No run history at")

    def test_clean_text_unchanged(self):
        assert us._redact_action("nothing sensitive") == "nothing sensitive"

    def test_redactor_failure_returns_text_unchanged(self, monkeypatch):
        """Mocked: ``ui_security._redact_paths`` raises; redaction must never crash a handler."""
        import backpropagate.ui_security as sec

        def boom(_t):
            raise RuntimeError("regex engine down")

        monkeypatch.setattr(sec, "_redact_paths", boom)
        assert us._redact_action("/home/alice/x") == "/home/alice/x"


# =============================================================================
# Numeric coercion / clamps
# =============================================================================


class TestCoerce:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(5, 5), (5.9, 5), ("7", 7), (" 8 ", 8), ("1e3", 1000), ("2.7", 2), (-3, -3),
         (True, None), (False, None), ("", None), ("  ", None), ("abc", None), ("1,5", None),
         (None, None), ([1], None), ({"a": 1}, None), (b"5", None)],
    )
    def test_coerce_int(self, raw, expected):
        assert us._coerce_int(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(5, 5.0), (0.25, 0.25), ("1e-3", 0.001), (" 2 ", 2.0), (True, None), ("", None),
         ("x", None), (None, None), ([1.0], None)],
    )
    def test_coerce_float(self, raw, expected):
        assert us._coerce_float(raw) == expected

    def test_clamp_int_branches(self):
        assert us._clamp_int("N", "5", 1, 10) == (5, "")
        assert us._clamp_int("N", 0, 1, 10) == (1, "N clamped to minimum 1 (was 0)")
        assert us._clamp_int("N", 99, 1, 10) == (10, "N clamped to maximum 10 (was 99)")
        value, err = us._clamp_int("N", "zzz", 1, 10)
        assert value is None and "must be an integer" in err and "'zzz'" in err

    def test_clamp_float_branches(self):
        assert us._clamp_float("F", "0.5", 0.0, 1.0) == (0.5, "")
        assert us._clamp_float("F", -1, 0.0, 1.0) == (0.0, "F clamped to minimum 0 (was -1)")
        assert us._clamp_float("F", 7, 0.0, 1.0) == (1.0, "F clamped to maximum 1 (was 7)")
        value, err = us._clamp_float("F", "zzz", 0.0, 1.0)
        assert value is None and "must be a number" in err


class TestApplyHelpers:
    @pytest.mark.parametrize(
        ("raw", "stored", "err_contains"),
        [("auto", "auto", ""), (" AUTO ", "auto", ""), ("8", "8", ""), (16, "16", ""),
         (0, "1", "minimum 1"), (-4, "1", "minimum 1"), (99999, "4096", "maximum 4096"),
         ("lots", None, "'auto' or a positive integer"), (None, None, "'auto'")],
    )
    def test_batch_size(self, raw, stored, err_contains):
        got, err = us._apply_batch_size(raw)
        assert got == stored and err_contains in err

    @pytest.mark.parametrize(
        ("raw", "stored", "err_contains"),
        [("", "", ""), ("   ", "", ""), ("q_proj", "q_proj", ""),
         ("q_proj,k_proj ,  v_proj", "q_proj, k_proj, v_proj", ""),
         ("q_proj; rm -rf /", None, "only letters"), ("1abc", None, "only letters"),
         ("a-b", None, "only letters"), ("$(id)", None, "only letters"),
         # a leading comma fails the identifier regex before the empty-list guard
         (",,,", None, "only letters"), (", ,", None, "only letters"),
         (",".join(f"m{i}" for i in range(33)), None, "At most 32"),
         (",".join(f"m{i}" for i in range(32)), ", ".join(f"m{i}" for i in range(32)), "")],
    )
    def test_target_modules(self, raw, stored, err_contains):
        got, err = us._apply_target_modules(raw)
        assert got == stored and err_contains in err

    @pytest.mark.parametrize(
        ("raw", "stored", "err_contains"),
        [("", "", ""), ("  ", "", ""), ("run-1.2_x", "run-1.2_x", ""), ("  ok  ", "ok", ""),
         ("a" * 129, None, "too long"), ("a" * 128, "a" * 128, ""),
         ("has space", None, "alphanumerics"), ("a/b", None, "alphanumerics"),
         ("ü", None, "alphanumerics")],
    )
    def test_wandb_run_name(self, raw, stored, err_contains):
        got, err = us._apply_wandb_run_name(raw)
        assert got == stored and err_contains in err


# =============================================================================
# AppState / AuthBadgeState
# =============================================================================


class TestAppState:
    def test_toggle_theme_round_trips(self):
        s = us.AppState()
        assert s.theme == "dark"
        s.toggle_theme()
        assert s.theme == "light"
        s.toggle_theme()
        assert s.theme == "dark"

    @pytest.mark.parametrize("surface", ["train", "multi-run", "export", "dataset"])
    def test_active_surface_accepts_known_values(self, surface):
        s = us.AppState()
        s.set_active_surface(surface)
        assert s.active_surface == surface

    @pytest.mark.parametrize("surface", ["settings", "", "TRAIN", "../admin", "runs"])
    def test_active_surface_ignores_unknown_values(self, surface):
        s = us.AppState()
        s.set_active_surface("export")
        s.set_active_surface(surface)
        assert s.active_surface == "export"


class TestAuthBadgeState:
    def _clear_env(self, monkeypatch):
        for k in ("BACKPROPAGATE_UI_AUTH", "BACKPROPAGATE_UI_SHARE_HOST",
                  "BACKPROPAGATE_UI_HOST_BIND", "BACKPROPAGATE_UI_LAUNCH_TOKEN",
                  "BACKPROPAGATE_UI_PORT"):
            monkeypatch.delenv(k, raising=False)

    def test_refresh_mirrors_the_context_without_the_password(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "alice:s3cret!")
        monkeypatch.setenv("BACKPROPAGATE_UI_PORT", "9000")
        s = us.AuthBadgeState()
        s.refresh()
        assert s.mode_key == "basic_local" and s.mode_color == "green"
        assert s.auth_user == "alice" and s.bind_port == "9000"
        assert s.reachable_from == "loopback-only"
        for value in (s.mode_text, s.hover_text, s.bind_host, s.auth_user):
            assert "s3cret" not in value

    def test_refresh_is_once_per_session(self, monkeypatch):
        self._clear_env(monkeypatch)
        s = us.AuthBadgeState()
        s.refresh()
        assert s.mode_key == "no_auth_local"
        monkeypatch.setenv("BACKPROPAGATE_UI_AUTH", "u:p")
        s.refresh()  # early exit: env drift is ignored for the lifetime of the session
        assert s.mode_key == "no_auth_local"

    def test_insecure_posture_is_flagged_red(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("BACKPROPAGATE_UI_SHARE_HOST", "x.trycloudflare.com")
        s = us.AuthBadgeState()
        s.refresh()
        assert (s.mode_key, s.mode_color) == ("insecure", "red")
        assert "NO AUTH" in s.hover_text


# =============================================================================
# TrainState
# =============================================================================


class TestTrainStateComputedVars:
    def test_loss_chart_data_shape(self):
        s = us.TrainState()
        assert s.loss_chart_data == []
        s.loss_history = [1.5, 1.0, 0.25]
        assert s.loss_chart_data == [
            {"step": 0, "loss": 1.5}, {"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.25}]

    @pytest.mark.parametrize(
        ("run_state", "complete"),
        [("idle", False), ("loading", False), ("active", False), ("paused", False),
         ("done", True), ("stopped", True), ("error", True)],
    )
    def test_run_complete(self, run_state, complete):
        s = us.TrainState()
        s.run_state = run_state
        assert s.run_complete is complete

    def test_recovery_messages_pick_the_most_recent_of_each_level(self):
        s = us.TrainState()
        s.events = [
            {"level": "ok", "msg": "ok-old"}, {"level": "warn", "msg": "warn-old"},
            {"level": "info", "msg": "info-old"}, {"level": "ok", "msg": "ok-new"},
            {"level": "err", "msg": "boom"}, {"level": "warn", "msg": "warn-new"},
            {"level": "info", "msg": "info-new"},
        ]
        assert s.latest_recovery_ok_msg == "ok-new"
        assert s.latest_recovery_warn_msg == "warn-new"
        assert s.latest_recovery_info_msg == "info-new"

    def test_recovery_messages_empty_when_absent_or_malformed(self):
        s = us.TrainState()
        assert (s.latest_recovery_ok_msg, s.latest_recovery_warn_msg,
                s.latest_recovery_info_msg) == ("", "", "")
        s.events = [{"level": "err", "msg": "x"}, "not-a-dict", {"level": "ok"}, {"msg": "no level"}]
        assert s.latest_recovery_ok_msg == ""  # ok entry with no msg -> ""
        assert s.latest_recovery_warn_msg == "" and s.latest_recovery_info_msg == ""


class TestTrainStateSettersExhaustive:
    @pytest.mark.parametrize(
        ("handler", "field", "good", "stored", "low", "clamped_low", "high", "clamped_high"),
        [
            ("set_steps", "steps", "250", 250, 0, 1, 10**9, 100_000),
            ("set_lora_r", "lora_r", 64, 64, 0, 1, 9999, 256),
            ("set_lora_alpha", "lora_alpha", "128", 128, -5, 1, 9999, 512),
            ("set_gpu_temp_threshold", "gpu_temp_threshold", 90, 90, 10, 50, 500, 105),
        ],
    )
    def test_int_setters(self, handler, field, good, stored, low, clamped_low, high, clamped_high):
        s = us.TrainState()
        err_field = f"{field}_error"
        getattr(s, handler)(good)
        assert getattr(s, field) == stored and getattr(s, err_field) == ""
        getattr(s, handler)(low)
        assert getattr(s, field) == clamped_low and "minimum" in getattr(s, err_field)
        getattr(s, handler)(high)
        assert getattr(s, field) == clamped_high and "maximum" in getattr(s, err_field)
        getattr(s, handler)("garbage")
        assert getattr(s, field) == clamped_high  # prior value kept
        assert "must be an integer" in getattr(s, err_field)

    @pytest.mark.parametrize(
        ("handler", "field", "good", "low", "clamped_low", "high", "clamped_high"),
        [
            ("set_learning_rate", "learning_rate", "1e-4", 0.0, 1e-7, 50, 1.0),
            ("set_lora_dropout", "lora_dropout", 0.1, -0.5, 0.0, 3, 1.0),
        ],
    )
    def test_float_setters(self, handler, field, good, low, clamped_low, high, clamped_high):
        s = us.TrainState()
        getattr(s, handler)(good)
        assert getattr(s, field) == float(good) and getattr(s, f"{field}_error") == ""
        getattr(s, handler)(low)
        assert getattr(s, field) == clamped_low and "minimum" in getattr(s, f"{field}_error")
        getattr(s, handler)(high)
        assert getattr(s, field) == clamped_high and "maximum" in getattr(s, f"{field}_error")
        getattr(s, handler)("x")
        assert getattr(s, field) == clamped_high and "must be a number" in getattr(s, f"{field}_error")

    def test_batch_size_setter(self):
        s = us.TrainState()
        s.set_batch_size("16")
        assert (s.batch_size, s.batch_size_error) == ("16", "")
        s.set_batch_size("lots")
        assert s.batch_size == "16" and "positive integer" in s.batch_size_error
        s.set_batch_size("0")
        assert s.batch_size == "1" and "minimum" in s.batch_size_error
        s.set_batch_size("auto")
        assert (s.batch_size, s.batch_size_error) == ("auto", "")

    def test_target_modules_setter_keeps_previous_on_rejection(self):
        s = us.TrainState()
        s.set_target_modules("q_proj,v_proj")
        assert s.target_modules == "q_proj, v_proj" and s.target_modules_error == ""
        s.set_target_modules("q_proj; DROP")
        assert s.target_modules == "q_proj, v_proj" and "only letters" in s.target_modules_error
        s.set_target_modules("")
        assert (s.target_modules, s.target_modules_error) == ("", "")

    def test_wandb_run_name_setter(self):
        s = us.TrainState()
        s.set_wandb_run_name("exp-1")
        assert (s.wandb_run_name, s.wandb_run_name_error) == ("exp-1", "")
        s.set_wandb_run_name("bad name!")
        assert s.wandb_run_name == "exp-1" and s.wandb_run_name_error
        s.set_wandb_run_name("")
        assert (s.wandb_run_name, s.wandb_run_name_error) == ("", "")

    def test_mode_and_flags(self):
        s = us.TrainState()
        s.set_train_mode("lora")
        assert s.train_mode == "lora"
        s.set_train_mode("2-bit")
        assert s.train_mode == "lora"
        s.set_gradient_checkpointing(0)
        assert s.gradient_checkpointing is False
        s.set_gradient_checkpointing(1)
        assert s.gradient_checkpointing is True


class TestTrainStateHandlers:
    def test_start_training_never_fakes_a_run(self):
        """ui-v2 P1: invalid input -> refusal on screen, never a fake spinner."""
        s = us.TrainState()
        s.start_training()
        s.start_training()
        assert s.run_state == "idle"
        assert s.job_refusal != ""  # dataset missing / outside sandbox
        assert s.job_id == ""

    @pytest.mark.parametrize("state", ["active", "loading", "paused"])
    def test_stop_training_resets_a_live_run(self, state):
        """No live child -> Stop quietly returns to idle (nothing to save)."""
        s = us.TrainState()
        s.run_state = state
        s.stop_training()
        assert s.run_state == "idle"

    def test_stop_training_requests_cooperative_stop(self, tmp_path, monkeypatch):
        """A live child gets control.json via the manager + an armed deadline."""
        import backpropagate.ui_jobs as ui_jobs

        calls: list[str] = []

        class _LiveManager:
            def is_alive(self, job_id):
                return True

            def grace_window(self, job_id):
                return 60.0

            def request_stop(self, job_id):
                calls.append(job_id)
                return True

        monkeypatch.setattr(ui_jobs, "get_job_manager", lambda: _LiveManager())
        s = us.TrainState()
        s.job_id = "run_live"
        s.run_state = "active"
        s.stop_training()
        assert calls == ["run_live"]
        assert s._stop_pending is True
        assert s._stop_deadline > 0
        assert s.run_state == "active"  # still running until the child exits
        assert any("Stop requested" in e["msg"] for e in s.events)

    @pytest.mark.parametrize("state", ["idle", "done", "error"])
    def test_stop_training_is_a_noop_when_nothing_is_running(self, state):
        s = us.TrainState()
        s.run_state = state
        s.stop_training()
        assert s.run_state == state and s.events == []


# =============================================================================
# MultiRunState
# =============================================================================


class TestMultiRunStateSetters:
    def test_shared_config_setters_match_train_state(self):
        s = us.MultiRunState()
        s.set_steps("250")
        s.set_batch_size("32")
        s.set_learning_rate("3e-4")
        s.set_lora_r(8)
        s.set_lora_alpha(16)
        s.set_lora_dropout("0.2")
        s.set_target_modules("q_proj")
        s.set_train_mode("lora")
        assert (s.steps, s.batch_size, s.learning_rate, s.lora_r, s.lora_alpha,
                s.lora_dropout, s.target_modules, s.train_mode) == (
            250, "32", 3e-4, 8, 16, 0.2, "q_proj", "lora")
        s.set_steps(0)
        s.set_learning_rate(9)
        s.set_lora_r("x")
        s.set_lora_alpha(10**6)
        s.set_lora_dropout(-1)
        s.set_batch_size("lots")
        s.set_target_modules("a b; c")
        s.set_train_mode("full")  # a multi-run merges adapters: never full
        assert s.steps == 1 and "minimum" in s.steps_error
        assert s.learning_rate == 1.0 and "maximum" in s.learning_rate_error
        assert s.lora_r == 8 and "integer" in s.lora_r_error
        assert s.lora_alpha == 512 and "maximum" in s.lora_alpha_error
        assert s.lora_dropout == 0.0 and "minimum" in s.lora_dropout_error
        assert s.batch_size == "32" and s.batch_size_error
        assert s.target_modules == "q_proj" and s.target_modules_error
        assert s.train_mode == "lora"

    @pytest.mark.parametrize(
        ("handler", "field", "good", "low", "clamped_low", "high", "clamped_high"),
        [
            ("set_num_runs", "num_runs", 5, 0, 1, 1000, 100),
            ("set_samples_per_run", "samples_per_run", "2000", -1, 1, 10**9, 1_000_000),
        ],
    )
    def test_sweep_shape_setters(self, handler, field, good, low, clamped_low, high, clamped_high):
        s = us.MultiRunState()
        getattr(s, handler)(good)
        assert getattr(s, field) == int(good) and getattr(s, f"{field}_error") == ""
        getattr(s, handler)(low)
        assert getattr(s, field) == clamped_low and "minimum" in getattr(s, f"{field}_error")
        getattr(s, handler)(high)
        assert getattr(s, field) == clamped_high and "maximum" in getattr(s, f"{field}_error")
        getattr(s, handler)("nope")
        assert getattr(s, field) == clamped_high and "integer" in getattr(s, f"{field}_error")

    def test_merge_mode_allowlist(self):
        s = us.MultiRunState()
        for mode in ("simple", "ties", "slao"):
            s.set_merge_mode(mode)
            assert s.merge_mode == mode
        s.set_merge_mode("evil")
        assert s.merge_mode == "slao"

    def test_replay_fraction_clamped_to_unit_interval(self):
        s = us.MultiRunState()
        s.set_replay_fraction("0.25")
        assert (s.replay_fraction, s.replay_fraction_error) == (0.25, "")
        s.set_replay_fraction(2)
        assert s.replay_fraction == 1.0 and "maximum" in s.replay_fraction_error
        s.set_replay_fraction(-2)
        assert s.replay_fraction == 0.0 and "minimum" in s.replay_fraction_error
        s.set_replay_fraction("abc")
        assert s.replay_fraction == 0.0 and "number" in s.replay_fraction_error

    def test_start_multi_run_hands_a_multi_run_job_to_the_job_state(self, sandbox):
        """ui-v2 P2: the Start button starts a real job (no CLI pointer)."""
        sandbox.out.mkdir(parents=True, exist_ok=True)
        data = sandbox.out / "data.jsonl"
        data.write_text('{"text": "hi"}\n')
        s = us.MultiRunState()
        s.set_dataset_path(str(data))
        s.set_num_runs(2)
        s.set_steps(15)
        s.set_merge_mode("ties")
        spec = s.start_multi_run()
        assert spec.handler.fn.__name__ == "start_job"
        payload = str(spec.args[0][1])
        for expected in ('"multi_run"', '"ties"', "data.jsonl"):
            assert expected in payload

    def test_start_multi_run_refuses_on_screen_when_a_field_is_invalid(self):
        s = us.MultiRunState()
        s.set_num_runs("nope")
        spec = s.start_multi_run()
        assert spec.handler.fn.__name__ == "refuse"
        assert "Fix the highlighted fields" in str(spec.args[0][1])


# =============================================================================
# ExportState
# =============================================================================


class TestExportStateSetters:
    def test_format_and_quant_allowlists(self):
        s = us.ExportState()
        for fmt in ("merged", "gguf", "lora"):
            s.set_format(fmt)
            assert s.format == fmt
        s.set_format("pickle")
        assert s.format == "lora"
        # ui-v2 P2: exactly the `backprop export --quantization` choices.
        for q in ("f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0", "q2_k"):
            s.set_gguf_quant(q)
            assert s.gguf_quant == q
        for rejected in ("q9_9", "q3_K_M", "q6_K"):
            s.set_gguf_quant(rejected)
            assert s.gguf_quant == "q2_k"

    @pytest.mark.parametrize(
        "name",
        ["llama3:8b", "my-model", "registry.io/org/model:tag", "a_b.c"],
    )
    def test_ollama_name_accepts_registry_style_names(self, name):
        s = us.ExportState()
        s.set_ollama_name(f"  {name}  ")
        assert (s.ollama_name, s.ollama_name_error) == (name, "")

    @pytest.mark.parametrize(
        "name",
        ["../escape", "a/../b", "/abs/path", "a\\b", "bad name", "x;rm", "a\x00b", "$(id)", "a|b"],
    )
    def test_ollama_name_rejects_path_like_and_injection_values(self, name):
        s = us.ExportState()
        s.set_ollama_name("good")
        s.set_ollama_name(name)
        assert s.ollama_name == "" and s.ollama_name_error == "Invalid Ollama model name"

    def test_ollama_name_cleared_with_empty_value(self):
        s = us.ExportState()
        s.set_ollama_name("good")
        s.set_ollama_name("")
        assert (s.ollama_name, s.ollama_name_error) == ("", "")

    def test_boolean_toggles(self):
        s = us.ExportState()
        s.set_ollama_register(1)
        s.set_hub_enabled("yes")
        s.set_hub_private(0)
        s.set_hub_include_base(True)
        assert (s.ollama_register, s.hub_enabled, s.hub_private, s.hub_include_base) == (
            True, True, False, True)

    def test_start_export_hands_an_export_job_to_the_job_state(self):
        """ui-v2 P2: Export starts a real job; Ollama only rides along on GGUF."""
        s = us.ExportState()
        s.set_format("gguf")
        s.set_gguf_quant("q8_0")
        s.set_ollama_register(True)
        s.set_ollama_name("my-model")
        spec = s.start_export()
        assert spec.handler.fn.__name__ == "start_job"
        payload = str(spec.args[0][1])
        for expected in ('"export"', '"gguf"', '"q8_0"', '"my-model"'):
            assert expected in payload
        s.set_format("lora")
        assert '"my-model"' not in str(s.start_export().args[0][1])

    def test_start_export_refuses_on_screen_when_a_field_is_invalid(self):
        s = us.ExportState()
        s.set_ollama_name("../escape")
        assert s.start_export().handler.fn.__name__ == "refuse"


class TestExportHubFieldValidation:
    @pytest.mark.parametrize("value", ["", "   "])
    def test_repo_id_blank_clears(self, value):
        s = us.ExportState()
        s.set_hub_repo_id("owner/repo")
        s.set_hub_repo_id(value)
        assert (s.hub_repo_id, s.hub_repo_id_error) == ("", "")

    def test_repo_id_valid(self):
        s = us.ExportState()
        s.set_hub_repo_id("  owner/my-repo.v2  ")
        assert (s.hub_repo_id, s.hub_repo_id_error) == ("owner/my-repo.v2", "")

    @pytest.mark.parametrize(
        ("value", "needle"),
        [("a" * 201, "too long"), ("no-slash", "<owner>/<repo>"),
         ("owner/repo name", "alnum"), ("owner/repo;rm", "alnum"), ("o/r\n", None)],
    )
    def test_repo_id_rejections_keep_previous_value(self, value, needle):
        s = us.ExportState()
        s.set_hub_repo_id("owner/ok")
        s.set_hub_repo_id(value)
        if needle:
            assert s.hub_repo_id == "owner/ok" and needle in s.hub_repo_id_error
        else:  # trailing newline is stripped, so it is simply valid
            assert s.hub_repo_id == "o/r"

    def test_branch_defaults_and_validation(self):
        s = us.ExportState()
        s.set_hub_branch("release/1.0")
        assert (s.hub_branch, s.hub_branch_error) == ("release/1.0", "")
        s.set_hub_branch("   ")
        assert (s.hub_branch, s.hub_branch_error) == ("main", "")
        s.set_hub_branch("x" * 101)
        assert s.hub_branch == "main" and "too long" in s.hub_branch_error
        s.set_hub_branch("bad branch")
        assert s.hub_branch == "main" and "alnum" in s.hub_branch_error

    def test_token_setter_is_write_only_and_length_checked(self):
        s = us.ExportState()
        s.set_hub_token("hf_" + "a" * 37)
        assert s.hub_token_set is True and s.hub_token_error == ""
        assert s._hub_token_value() == "hf_" + "a" * 37
        s.set_hub_token("short")
        assert s.hub_token_set is False and "20-200" in s.hub_token_error
        s.set_hub_token("x" * 201)
        assert s.hub_token_set is False and s.hub_token_error
        s.set_hub_token("hf_" + "b" * 37)
        s.set_hub_token("   ")
        assert (s._hub_token_value(), s.hub_token_set, s.hub_token_error) == ("", False, "")
        s.set_hub_token(None)  # type: ignore[arg-type]
        assert s.hub_token_set is False

    def test_the_raw_token_never_lands_in_a_public_field_or_error(self):
        s = us.ExportState()
        secret = "hf_SECRETSECRETSECRETSECRETSECRET1234"
        s.set_hub_token(secret)
        public = {k: str(getattr(s, k)) for k in s.get_fields() if not k.startswith("_")}
        assert all(secret not in v for v in public.values())

    def test_token_file_path_setter(self, tmp_path):
        s = us.ExportState()
        f = tmp_path / "hf-token"
        f.write_text("hf_" + "a" * 37, encoding="utf-8")
        s.set_hub_token_file_path(str(f))
        assert Path(s.hub_token_file_path) == f.resolve() and s.hub_token_file_path_error == ""
        s.set_hub_token_file_path(str(tmp_path / "missing"))
        assert s.hub_token_file_path == "" and "does not exist" in s.hub_token_file_path_error
        s.set_hub_token_file_path("  ")
        assert (s.hub_token_file_path, s.hub_token_file_path_error) == ("", "")


class _PushRecorder:
    def __init__(self, exc=None):
        self.calls = []
        self.exc = exc

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc


@pytest.fixture
def push_recorder(monkeypatch):
    """Mocked: ``backpropagate.export.push_to_hub`` (the HF Hub network call)."""
    import backpropagate.export as export_mod

    rec = _PushRecorder()
    monkeypatch.setattr(export_mod, "push_to_hub", rec)
    return rec


def _ready_export_state(sandbox, **over):
    s = us.ExportState()
    s.set_source_model_path(str(sandbox.out / "adapter"))
    s.set_hub_repo_id("owner/repo")
    s.set_hub_token("hf_" + "a" * 37)
    for k, v in over.items():
        setattr(s, k, v)
    return s


class TestPushToHubPreflight:
    def test_requires_a_source_path(self, push_recorder):
        s = us.ExportState()
        s.push_to_hub()
        assert (s.hub_status, push_recorder.calls) == ("error", [])
        assert "source adapter" in s.hub_message

    def test_requires_a_valid_repo_id(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox, hub_repo_id="")
        s.push_to_hub()
        assert s.hub_status == "error" and "repo id" in s.hub_message and not push_recorder.calls
        s = _ready_export_state(sandbox, hub_repo_id_error="bad")
        s.push_to_hub()
        assert s.hub_status == "error" and not push_recorder.calls

    def test_requires_some_token(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox)
        s.set_hub_token("")
        s.push_to_hub()
        assert s.hub_status == "error" and "API token" in s.hub_message
        assert push_recorder.calls == []

    def test_inline_token_and_token_file_are_mutually_exclusive(self, sandbox, tmp_path, push_recorder):
        s = _ready_export_state(sandbox)
        f = tmp_path / "tok"
        f.write_text("hf_" + "z" * 37, encoding="utf-8")
        s.set_hub_token_file_path(str(f))
        s.push_to_hub()
        assert s.hub_status == "error" and "mutually exclusive" in s.hub_message
        assert push_recorder.calls == []

    def test_branch_error_blocks_push(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox, hub_branch_error="bad branch")
        s.push_to_hub()
        assert s.hub_status == "error" and "branch" in s.hub_message.lower()
        assert push_recorder.calls == []

    def test_token_file_error_blocks_push(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox, hub_token_file_path_error="bad file")
        s.set_hub_token("")  # inline token cleared; only the (errored) file path remains
        s.hub_token_file_path = "/some/path"
        s.push_to_hub()
        assert s.hub_status == "error" and push_recorder.calls == []

    def test_token_file_error_blocks_push_even_with_inline_token(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox, hub_token_file_path_error="bad file")
        s.push_to_hub()
        assert s.hub_status == "error" and "Token-file path error" in s.hub_message
        assert push_recorder.calls == []


class TestPushToHubExecution:
    def test_inline_token_push_passes_exact_arguments_and_wipes_the_token(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox, hub_private=False, hub_include_base=True)
        s.set_hub_branch("dev")
        s.push_to_hub()
        assert len(push_recorder.calls) == 1
        call = push_recorder.calls[0]
        assert call == {
            "local_path": s.source_model_path, "repo_id": "owner/repo",
            "token": "hf_" + "a" * 37, "private": False, "revision": "dev",
            "include_base": True,
        }
        assert s.hub_status == "done"
        assert "huggingface.co/owner/repo" in s.hub_message and "dev" in s.hub_message
        # credential hygiene: wiped after use, and never echoed in the message
        assert s._hub_token_value() == "" and s.hub_token_set is False
        assert "a" * 37 not in s.hub_message

    def test_token_file_push_reads_the_file_at_push_time(self, sandbox, tmp_path, push_recorder):
        f = tmp_path / "hf-token"
        f.write_text("hf_" + "f" * 37 + "\n", encoding="utf-8")
        s = _ready_export_state(sandbox)
        s.set_hub_token("")
        s.set_hub_token_file_path(str(f))
        s.push_to_hub()
        assert push_recorder.calls[0]["token"] == "hf_" + "f" * 37  # stripped
        assert s.hub_status == "done"
        assert s.hub_token_file_path  # path kept so a retry needs no re-entry

    def test_empty_token_file_surfaces_a_sanitised_error_without_pushing(
        self, sandbox, tmp_path, push_recorder
    ):
        f = tmp_path / "empty"
        f.write_text("", encoding="utf-8")
        s = _ready_export_state(sandbox)
        s.set_hub_token("")
        s.set_hub_token_file_path(str(f))
        s.push_to_hub()
        assert s.hub_status == "error" and "empty" in s.hub_message
        assert push_recorder.calls == []

    def test_backpropagate_error_message_is_surfaced_with_code_and_hint(self, sandbox, push_recorder):
        push_recorder.exc = BackpropagateError(
            message="upload refused for /home/alice/adapter", code="RUNTIME_HUB_PUSH",
            suggestion="check your token scope")
        s = _ready_export_state(sandbox)
        s.push_to_hub()
        assert s.hub_status == "error"
        assert s.hub_message.startswith("RUNTIME_HUB_PUSH: ")
        assert "alice" not in s.hub_message and "Try: check your token scope" in s.hub_message
        assert s._hub_token_value()  # kept on failure so the operator can retry

    def test_foreign_exception_is_opaque_to_the_client(self, sandbox, push_recorder):
        push_recorder.exc = RuntimeError("401 for token hf_" + "q" * 37 + " at /home/alice/x")
        s = _ready_export_state(sandbox)
        s.push_to_hub()
        assert s.hub_status == "error"
        assert "Internal error during pushing to HuggingFace Hub" in s.hub_message
        assert "alice" not in s.hub_message and "q" * 37 not in s.hub_message

    def test_sanitiser_failure_falls_back_to_a_redacted_trimmed_message(
        self, sandbox, push_recorder, monkeypatch
    ):
        """Mocked: ``ui_security.sanitize_error_for_user`` itself raises."""
        import backpropagate.ui_security as sec

        def boom(*a, **k):
            raise RuntimeError("sanitiser down")

        monkeypatch.setattr(sec, "sanitize_error_for_user", boom)
        push_recorder.exc = OSError("cannot read /home/alice/secret/model.bin " + "x" * 400)
        s = _ready_export_state(sandbox)
        s.push_to_hub()
        assert s.hub_status == "error" and s.hub_message.startswith("Push failed: OSError")
        assert "alice" not in s.hub_message and len(s.hub_message) < 300

    def test_clear_hub_status(self, sandbox, push_recorder):
        s = _ready_export_state(sandbox)
        s.push_to_hub()
        s.clear_hub_status()
        assert (s.hub_status, s.hub_message) == ("", "")


# =============================================================================
# DatasetState (non-upload handlers)
# =============================================================================


class TestDatasetStateSimpleHandlers:
    def test_format_hint_allowlist(self):
        s = us.DatasetState()
        for hint in ("sharegpt", "alpaca", "openai", "jsonl", "auto"):
            s.set_format_hint(hint)
            assert s.format_hint == hint
        s.set_format_hint("yaml")
        assert s.format_hint == "auto"

    def test_filter_toggles_coerce_to_bool(self):
        s = us.DatasetState()
        s.set_dedup_enabled(0)
        s.set_drop_empty("")
        s.set_apply_curriculum("x")
        assert (s.dedup_enabled, s.drop_empty, s.apply_curriculum) == (False, False, True)

    def test_min_tokens_pushes_max_tokens_up(self):
        s = us.DatasetState()
        s.set_max_tokens(100)
        s.set_min_tokens(500)
        assert s.min_tokens == 500 and s.max_tokens == 500 and s.min_tokens_error == ""

    def test_min_tokens_below_max_leaves_max_alone(self):
        s = us.DatasetState()
        s.set_max_tokens(2048)
        s.set_min_tokens(10)
        assert s.max_tokens == 2048

    def test_max_tokens_below_min_is_raised_to_match_and_says_so(self):
        s = us.DatasetState()
        s.set_min_tokens(500)
        s.set_max_tokens(100)
        assert s.max_tokens == 500 and "raised to match" in s.max_tokens_error
        s.set_max_tokens(0)  # no limit is always allowed
        assert s.max_tokens == 0 and s.max_tokens_error == ""

    def test_token_bounds_clamped_and_garbage_keeps_previous(self):
        s = us.DatasetState()
        s.set_min_tokens(-5)
        assert s.min_tokens == 0 and "minimum" in s.min_tokens_error
        s.set_max_tokens(10**9)
        assert s.max_tokens == 1_000_000 and "maximum" in s.max_tokens_error
        s.set_max_tokens("junk")
        assert s.max_tokens == 1_000_000 and "integer" in s.max_tokens_error
        s.set_min_tokens("junk")
        assert s.min_tokens == 0 and "integer" in s.min_tokens_error

    def test_basename_and_has_upload_never_expose_the_directory(self):
        s = us.DatasetState()
        assert (s.uploaded_basename, s.has_upload) == ("", False)
        s._uploaded_path = "/home/alice/.backpropagate/ui-outputs/uploads/data.jsonl"
        assert s.uploaded_basename == "data.jsonl" and s.has_upload is True
