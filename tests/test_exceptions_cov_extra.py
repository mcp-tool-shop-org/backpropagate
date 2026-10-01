"""Coverage tests for ``backpropagate.exceptions`` and ``backpropagate.security``.

Pure logic, nothing mocked except the ``safetensors`` import (simulating the
extra being absent) and the module lock in the double-checked-locking test.
"""

from __future__ import annotations

import logging
import sys

import pytest
import torch

from backpropagate import security
from backpropagate.exceptions import (
    ERROR_CODES,
    BackpropagateError,
    FsdpUnavailableError,
    FullFinetuneModelTooLargeError,
    MLXUnavailableError,
    PartialSuccess,
)

# =============================================================================
# BackpropagateError.to_dict
# =============================================================================


class TestToDictEnvelope:
    def test_minimal_envelope_omits_optional_keys(self):
        err = BackpropagateError("plain", code="RUNTIME_TRAINING_FAILED")
        assert err.to_dict() == {
            "type": "BackpropagateError",
            "code": "RUNTIME_TRAINING_FAILED",
            "message": "plain",
            "retryable": False,
        }

    def test_full_envelope_includes_suggestion_details_and_cause(self):
        cause = OSError("disk full")
        err = BackpropagateError(
            "boom", details={"path": "/x"}, suggestion="free space", code="RUNTIME_TRAINING_FAILED",
            cause=cause, retryable=True,
        )
        env = err.to_dict()
        assert env["suggestion"] == "free space"
        assert env["details"] == {"path": "/x"}
        assert env["cause"] == "OSError: disk full"
        assert env["retryable"] is True
        assert err.__cause__ is cause
        assert str(err) == "boom" and repr(err) == "BackpropagateError('boom')"
        assert err.args[0] == "boom\n\nSuggestion: free space"

    def test_missing_code_stays_none_and_is_not_invented(self):
        assert BackpropagateError("x").code is None
        assert BackpropagateError("x").to_dict()["code"] is None

    def test_unknown_code_warns_but_is_preserved(self, caplog):
        with caplog.at_level(logging.WARNING, logger="backpropagate.exceptions"):
            err = BackpropagateError("x", code="RUNTIME_GPU_OOO")
        assert err.code == "RUNTIME_GPU_OOO"
        assert any("unknown code='RUNTIME_GPU_OOO'" in r.getMessage() for r in caplog.records)
        assert "RUNTIME_GPU_OOO" not in ERROR_CODES


# =============================================================================
# FullFinetuneModelTooLargeError
# =============================================================================


class TestFullFinetuneModelTooLarge:
    def test_exact_count_named_in_message_and_details(self):
        err = FullFinetuneModelTooLargeError("org/big", param_count_billions=7.6, ceiling_billions=4.0)
        assert "approximately 7.6B parameters" in err.message
        assert "mode='full' supports models up to 4.0B parameters on this card" in err.message
        assert "mode='lora'" in err.message
        assert err.code == "RUNTIME_FULL_FT_MODEL_TOO_LARGE" and err.retryable is False
        assert err.details == {
            "model_name": "org/big", "ceiling_billions": 4.0, "offload_recoverable": False,
            "offload_active": False, "param_count_billions": 7.6,
        }

    def test_unknown_count_uses_the_generic_phrase_and_omits_the_detail(self):
        err = FullFinetuneModelTooLargeError("org/mystery")
        assert "exceeds the documented parameter ceiling" in err.message
        assert "param_count_billions" not in err.details
        assert "offload_ceiling_billions" not in err.details

    def test_offload_recoverable_points_at_the_offload_flag_before_lora(self):
        err = FullFinetuneModelTooLargeError(
            "org/mid", param_count_billions=9.0, ceiling_billions=4.0,
            offload_ceiling_billions=14.0, offload_recoverable=True,
        )
        assert "--full-ft-offload" in err.message and "~14.0B" in err.message
        assert err.message.index("--full-ft-offload") < err.message.index("mode='lora'")
        assert err.details["offload_ceiling_billions"] == 14.0 and err.details["offload_recoverable"] is True

    def test_offload_already_active_says_even_with_offload(self):
        err = FullFinetuneModelTooLargeError(
            "org/huge", param_count_billions=30.0, ceiling_billions=14.0,
            offload_ceiling_billions=14.0, offload_active=True, suggestion="use a smaller model",
        )
        assert "14.0B parameters even with FSDP2 CPU-offload" in err.message
        assert "--full-ft-offload" not in err.message
        assert err.suggestion == "use a smaller model"
        assert err.details["offload_active"] is True

    def test_recoverable_without_a_ceiling_falls_back_to_the_lora_message(self):
        err = FullFinetuneModelTooLargeError("m", offload_recoverable=True)
        assert "--full-ft-offload" not in err.message


# =============================================================================
# MLX / FSDP availability errors, PartialSuccess
# =============================================================================


class TestAvailabilityErrors:
    def test_mlx_without_reason(self):
        err = MLXUnavailableError()
        assert err.code == "DEP_MLX_UNAVAILABLE" and err.retryable is False
        assert err.reason is None and err.details == {}
        assert err.message.endswith("(mlx-lm is macOS + arm64 ONLY).")
        assert "backpropagate[mlx]" in err.suggestion

    def test_mlx_with_reason_and_custom_suggestion(self):
        err = MLXUnavailableError("import mlx_lm failed", suggestion="use CUDA")
        assert err.message.endswith("import mlx_lm failed")
        assert err.details == {"reason": "import mlx_lm failed"}
        assert err.suggestion == "use CUDA"

    def test_fsdp_without_reason(self):
        err = FsdpUnavailableError()
        assert err.code == "DEP_FSDP_UNAVAILABLE" and err.details == {}
        assert "full_ft_offload=True" in err.message
        assert "accelerate" in err.suggestion

    def test_fsdp_with_reason(self):
        err = FsdpUnavailableError("torch too old", suggestion="upgrade torch")
        assert err.message.endswith("torch too old")
        assert err.details == {"reason": "torch too old"} and err.suggestion == "upgrade torch"


class TestPartialSuccess:
    def test_counts_and_merged_details(self):
        err = PartialSuccess(
            "8 of 10 exported", total_items=10, succeeded=8, failed=2,
            suggestion="retry the failed two", details={"failed_items": ["a", "b"]},
        )
        assert err.code == "PARTIAL_SUCCESS" and err.retryable is False
        assert (err.total_items, err.succeeded, err.failed) == (10, 8, 2)
        assert err.details == {"total_items": 10, "succeeded": 8, "failed": 2, "failed_items": ["a", "b"]}
        assert err.to_dict()["suggestion"] == "retry the failed two"

    def test_details_default_to_the_counts_only(self):
        err = PartialSuccess("x", 3, 1, 2)
        assert err.details == {"total_items": 3, "succeeded": 1, "failed": 2}


# =============================================================================
# security.py
# =============================================================================


class TestSafeTorchLoadFallbacks:
    def test_safetensors_missing_falls_back_to_weights_only_torch_load(self, tmp_path, monkeypatch, caplog):
        path = tmp_path / "weights.safetensors"
        torch.save({"w": torch.arange(4.0)}, path)  # a .safetensors name, pickle payload
        monkeypatch.setitem(sys.modules, "safetensors.torch", None)
        with caplog.at_level(logging.WARNING, logger="backpropagate.security"):
            state = security.safe_torch_load(path)
        assert torch.equal(state["w"], torch.arange(4.0))
        assert any("safetensors not installed" in r.getMessage() for r in caplog.records)

    def test_weights_only_false_warns_loudly_but_still_loads(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        torch.save({"w": torch.ones(2)}, path)
        with pytest.warns(security.SecurityWarning, match="defeats the weights_only"):
            state = security.safe_torch_load(path, weights_only=False)
        assert torch.equal(state["w"], torch.ones(2))

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            security.safe_torch_load(tmp_path / "gone.pt")


class TestTorchSecurityCheckOnce:
    def test_second_thread_that_lost_the_race_does_not_recheck(self, monkeypatch):
        """Double-checked locking: the flag flips while a caller waits on the lock."""
        calls = []
        monkeypatch.setattr(security, "check_torch_security", lambda: calls.append(1))
        monkeypatch.setattr(security, "_torch_security_checked", False)

        class WinsTheRace:
            def __enter__(self):
                security._torch_security_checked = True  # another thread finished first
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(security, "_torch_security_lock", WinsTheRace())
        security._ensure_torch_security_checked()
        assert calls == []

    def test_check_runs_exactly_once_then_short_circuits(self, monkeypatch):
        calls = []
        monkeypatch.setattr(security, "check_torch_security", lambda: calls.append(1))
        monkeypatch.setattr(security, "_torch_security_checked", False)
        security._ensure_torch_security_checked()
        security._ensure_torch_security_checked()
        assert calls == [1] and security._torch_security_checked is True


class TestAuditLog:
    def test_success_is_info_with_all_context_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="backpropagate.security.audit"):
            security.audit_log("model_load", path="/m", user="mike", details={"size": 7})
        (rec,) = [r for r in caplog.records if r.name == "backpropagate.security.audit"]
        assert rec.levelno == logging.INFO and rec.getMessage() == "AUDIT: model_load"
        assert (rec.operation, rec.success, rec.path, rec.user, rec.size) == ("model_load", True, "/m", "mike", 7)

    def test_failure_is_a_warning_and_omits_absent_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="backpropagate.security.audit"):
            security.audit_log("export", success=False)
        (rec,) = [r for r in caplog.records if r.name == "backpropagate.security.audit"]
        assert rec.levelno == logging.WARNING and rec.success is False
        assert not hasattr(rec, "path") and not hasattr(rec, "user")

