"""Coverage tests for the argparse type validators and print helpers in cli.py.

Nothing is mocked: every validator is a pure function and every print helper
writes to the real (captured) stdout/stderr.
"""

from __future__ import annotations

import argparse

import pytest

from backpropagate import cli
from backpropagate.exceptions import BackpropagateError


class TestIntValidators:
    def test_positive_int_accepts_and_rejects(self):
        assert cli._positive_int("3") == 3
        with pytest.raises(argparse.ArgumentTypeError, match="expected an integer"):
            cli._positive_int("abc")
        with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
            cli._positive_int("0")

    def test_auto_or_positive_int(self):
        assert cli._auto_or_positive_int(" AUTO ") == "auto"
        assert cli._auto_or_positive_int("4") == 4
        with pytest.raises(argparse.ArgumentTypeError, match="'auto' or a positive integer"):
            cli._auto_or_positive_int("foo")
        with pytest.raises(argparse.ArgumentTypeError, match="'auto' or a positive integer"):
            cli._auto_or_positive_int("0")

    def test_non_negative_int(self):
        assert cli._non_negative_int("0") == 0
        assert cli._non_negative_int("7") == 7
        with pytest.raises(argparse.ArgumentTypeError, match="expected an integer"):
            cli._non_negative_int("x")
        with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
            cli._non_negative_int("-1")

    def test_port_int(self):
        assert cli._port_int("8080") == 8080
        assert cli._port_int("1") == 1
        assert cli._port_int("65535") == 65535
        with pytest.raises(argparse.ArgumentTypeError, match="expected an integer port"):
            cli._port_int("http")
        for bad in ("0", "65536", "-5"):
            with pytest.raises(argparse.ArgumentTypeError, match="1..65535"):
                cli._port_int(bad)


class TestFloatValidators:
    def test_positive_float(self):
        assert cli._positive_float("2e-4") == pytest.approx(2e-4)
        with pytest.raises(argparse.ArgumentTypeError, match="expected a float"):
            cli._positive_float("nope")
        with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
            cli._positive_float("0")
        with pytest.raises(argparse.ArgumentTypeError, match="must be positive"):
            cli._positive_float("-1.5")

    def test_unit_float(self):
        assert cli._unit_float("0") == 0.0
        assert cli._unit_float("1") == 1.0
        assert cli._unit_float("0.5") == 0.5
        with pytest.raises(argparse.ArgumentTypeError, match="expected a float"):
            cli._unit_float("half")
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0, 1\]"):
            cli._unit_float("90")
        with pytest.raises(argparse.ArgumentTypeError, match=r"\[0, 1\]"):
            cli._unit_float("-0.1")


class TestHostStr:
    @pytest.mark.parametrize(
        "value",
        ["127.0.0.1", "0.0.0.0", "::1", "[::1]", "localhost", "my-box.example.com", "host.example."],
    )
    def test_accepts_valid(self, value):
        assert cli._host_str(value) == value

    def test_rejects_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
            cli._host_str("")

    @pytest.mark.parametrize("value", [" localhost", "local host", "a\tb"])
    def test_rejects_whitespace(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="whitespace"):
            cli._host_str(value)

    def test_rejects_leading_dash(self):
        with pytest.raises(argparse.ArgumentTypeError, match="start with '-'"):
            cli._host_str("-0.0.0.0")

    def test_rejects_overlong_hostname(self):
        with pytest.raises(argparse.ArgumentTypeError, match="invalid hostname"):
            cli._host_str("a" * 254)

    def test_rejects_lone_dot(self):
        # trailing-dot stripping leaves an empty hostname
        with pytest.raises(argparse.ArgumentTypeError, match="invalid hostname"):
            cli._host_str(".")

    @pytest.mark.parametrize("value", ["bad_host", "bad-.example", "a..b", "x" * 64])
    def test_rejects_bad_label(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="invalid"):
            cli._host_str(value)


class TestAuthCredential:
    def test_valid_returns_original(self):
        assert cli._auth_credential("alice:s3cr:et") == "alice:s3cr:et"

    def test_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Got an empty value"):
            cli._auth_credential("")

    def test_no_colon(self):
        with pytest.raises(argparse.ArgumentTypeError, match="no colon separator"):
            cli._auth_credential("alicepass")

    def test_empty_username(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty username"):
            cli._auth_credential(":pw")

    def test_empty_password(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty password"):
            cli._auth_credential("alice:")

    def test_username_with_space(self):
        with pytest.raises(argparse.ArgumentTypeError, match="username contains a forbidden"):
            cli._auth_credential("al ice:pw")

    def test_password_with_newline(self):
        with pytest.raises(argparse.ArgumentTypeError, match="password contains a forbidden"):
            cli._auth_credential("alice:pw\nx")


class TestPrintHelpers:
    def test_print_structured_error_with_code_and_run_id(self, capsys):
        exc = BackpropagateError("boom", details={"run_id": "abc123"})
        exc.code = "RUNTIME_GPU_OOM"
        cli._print_structured_error(exc, prefix="Train: ")
        captured = capsys.readouterr()
        assert "[RUNTIME_GPU_OOM] Train: boom" in captured.err
        assert "Run id: abc123" in captured.out

    def test_print_structured_error_without_code_or_details(self, capsys):
        class Plain:
            message = "plain failure"

        cli._print_structured_error(Plain())
        captured = capsys.readouterr()
        assert "plain failure" in captured.err
        assert "[" not in captured.err.replace("[ERROR]", "")
        assert "Run id" not in captured.out

    def test_print_error_redacted_scrubs_secret(self, capsys):
        cli._print_error_redacted(RuntimeError("Authorization: Bearer abcdef1234567890"), prefix="x: ")
        err = capsys.readouterr().err
        assert "x: RuntimeError:" in err
        assert "abcdef1234567890" not in err

    def test_progress_bar_update_and_finish(self, capsys):
        bar = cli.ProgressBar(total=4, width=8, prefix="P ")
        bar.update(2, suffix="half")
        out = capsys.readouterr().out
        assert "P [####----]  50.0% half" in out
        bar.finish()
        out = capsys.readouterr().out
        assert "100.0%" in out and out.endswith("\n")
        assert bar.current == 4

    def test_progress_bar_zero_total(self, capsys):
        bar = cli.ProgressBar(total=0, width=4)
        bar.update(0)
        assert "0.0%" in capsys.readouterr().out

    def test_print_header_kv_warning_info_success(self, capsys):
        cli._print_header("Title")
        cli._print_kv("k", "v", indent=4)
        cli._print_warning("careful")
        cli._print_info("fyi")
        cli._print_success("yay")
        out = capsys.readouterr().out
        assert "Title" in out and "-----" in out
        assert "    " in out and "k:" in out and "v" in out
        assert "[WARN]" in out and "careful" in out
        assert "[INFO]" in out and "fyi" in out
        assert "[OK]" in out and "yay" in out

    def test_reconfigure_stdio_swallows_failures(self, monkeypatch):
        class Bad:
            def reconfigure(self, **kw):
                raise ValueError("detached")

        class NoReconf:
            pass

        monkeypatch.setattr(cli.sys, "stdout", Bad())
        monkeypatch.setattr(cli.sys, "stderr", NoReconf())
        cli._reconfigure_stdio_utf8()  # must not raise
