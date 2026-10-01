"""Coverage tests for ``cmd_validate`` in cli.py.

Nothing is mocked: real JSONL files on ``tmp_path`` go through the real
``validate_dataset`` / ``detect_format`` (torch-free).
"""

from __future__ import annotations

import json

import pytest

from backpropagate import cli
from tests.helpers.cli_cov_support import last_json, parse

GOOD = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}
BAD_FORMAT = {"unexpected": 1}
WARN_ROLE = {"messages": [{"role": "wizard", "content": "hi"}, {"role": "assistant", "content": "yo"}]}


def _write(tmp_path, lines, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _j(row):
    return json.dumps(row)


class TestValidateHuman:
    def test_clean_dataset_passes(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD), "", _j(GOOD)])
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Lines scanned: 3" in out and "Samples parsed: 2" in out
        assert "Format hint: auto" in out and "Format detected: sharegpt" in out
        assert "Total rows: 2" in out and "Valid rows: 2" in out
        assert "Dataset validation PASSED" in out

    def test_validation_errors_fail_with_65(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD), _j(BAD_FORMAT)])
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_DATA_ERR
        out = capsys.readouterr()
        assert "Validation errors" in out.out and "row 1: [unknown_format]" in out.out
        assert "Dataset validation FAILED" in out.err

    def test_error_listing_is_truncated_at_ten(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(BAD_FORMAT)] * 15)
        assert cli.cmd_validate(parse(["validate", str(p), "--max-errors", "50"])) == cli.EXIT_DATA_ERR
        assert "... and 5 more" in capsys.readouterr().out

    def test_warnings_do_not_fail(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(WARN_ROLE)])
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Warnings" in out and "[invalid_role]" in out
        assert "Dataset validation PASSED" in out

    def test_warning_listing_is_truncated_at_ten(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(WARN_ROLE)] * 13)
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_OK
        assert "... and 3 more" in capsys.readouterr().out

    def test_parse_errors_fail_even_if_rows_are_valid(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD), "{not json", _j(GOOD)])
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_DATA_ERR
        out = capsys.readouterr()
        assert "1 JSON parse error(s)" in out.out
        assert "JSON parse errors" in out.out and "line 2:" in out.out
        assert "validation FAILED" in out.err

    def test_parse_error_listing_truncated_and_max_errors_stops_scan(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD)] + ["{bad"] * 30)
        assert cli.cmd_validate(parse(["validate", str(p), "--max-errors", "12"])) == cli.EXIT_DATA_ERR
        out = capsys.readouterr().out
        assert "Lines scanned: 13" in out  # scan stopped at the max-errors threshold
        assert "... and 2 more" in out

    def test_no_parseable_rows(self, tmp_path, capsys):
        p = _write(tmp_path, ["{bad", "also bad"])
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_DATA_ERR
        assert "Dataset has no parseable rows." in capsys.readouterr().err

    def test_max_samples_limits_what_is_read(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD)] * 10)
        assert cli.cmd_validate(parse(["validate", str(p), "--max-samples", "3"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Samples parsed: 3" in out and "Lines scanned: 3" in out

    def test_explicit_format_hint(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD)])
        assert cli.cmd_validate(parse(["validate", str(p), "--format", "sharegpt"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Format hint: sharegpt" in out and "Format detected: sharegpt" in out

    def test_unknown_format_hint(self, tmp_path, capsys):
        p = _write(tmp_path, [_j(GOOD)])
        args = parse(["validate", str(p)])
        args.format = "klingon"
        assert cli.cmd_validate(args) == cli.EXIT_USER_ERROR
        assert "Unknown format hint: klingon" in capsys.readouterr().err

    def test_missing_and_directory(self, tmp_path, capsys):
        assert cli.cmd_validate(parse(["validate", str(tmp_path / "nope.jsonl")])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Dataset not found" in captured.err and "backprop train --data" in captured.out
        assert cli.cmd_validate(parse(["validate", str(tmp_path)])) == cli.EXIT_USER_ERROR
        assert "is a directory" in capsys.readouterr().err

    def test_not_utf8(self, tmp_path, capsys):
        p = tmp_path / "latin.jsonl"
        p.write_bytes(b'{"text": "caf\xe9"}\n')
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "not valid UTF-8" in captured.err and "iconv" in captured.out

    def test_unreadable(self, tmp_path, monkeypatch, capsys):
        p = _write(tmp_path, [_j(GOOD)])
        import builtins

        real_open = builtins.open

        def deny(path, *a, **k):
            if str(path) == str(p):
                raise PermissionError("denied")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        assert cli.cmd_validate(parse(["validate", str(p)])) == cli.EXIT_USER_ERROR
        assert "Could not read dataset: denied" in capsys.readouterr().err


class TestValidateJson:
    def _run(self, capsys, path, *extra, code):
        args = parse(["validate", str(path), "--json", *extra])
        assert cli.cmd_validate(args) == code
        return last_json(capsys.readouterr().out)

    def test_clean(self, tmp_path, capsys):
        payload = self._run(capsys, _write(tmp_path, [_j(GOOD)]), code=cli.EXIT_OK)
        assert payload["is_valid"] is True and payload["format_detected"] == "sharegpt"
        assert payload["total_rows"] == 1 and payload["errors"] == [] and payload["parse_errors"] == []

    def test_errors_and_warnings_listed(self, tmp_path, capsys):
        payload = self._run(capsys, _write(tmp_path, [_j(BAD_FORMAT), _j(WARN_ROLE)]), code=cli.EXIT_DATA_ERR)
        assert payload["is_valid"] is False
        assert payload["errors"][0]["error_type"] == "unknown_format"
        assert payload["warnings"][0]["error_type"] == "invalid_role"

    def test_parse_errors_make_it_invalid(self, tmp_path, capsys):
        payload = self._run(capsys, _write(tmp_path, [_j(GOOD), "{bad"]), code=cli.EXIT_DATA_ERR)
        assert payload["is_valid"] is False
        assert payload["parse_errors"][0]["line"] == 2

    @pytest.mark.parametrize("scenario", ["missing", "dir", "empty", "badhint", "utf8"])
    def test_error_payloads(self, tmp_path, capsys, scenario):
        expected = {
            "missing": ("dataset_not_found", cli.EXIT_USER_ERROR),
            "dir": ("dataset_path_is_directory", cli.EXIT_USER_ERROR),
            "empty": ("no_parseable_rows", cli.EXIT_DATA_ERR),
            "badhint": ("unknown_format_hint", cli.EXIT_USER_ERROR),
            "utf8": ("not_utf8", cli.EXIT_USER_ERROR),
        }[scenario]
        if scenario == "missing":
            path = tmp_path / "nope.jsonl"
        elif scenario == "dir":
            path = tmp_path
        elif scenario == "empty":
            path = _write(tmp_path, ["{bad"])
        elif scenario == "utf8":
            path = tmp_path / "latin.jsonl"
            path.write_bytes(b'{"text": "caf\xe9"}\n')
        else:
            path = _write(tmp_path, [_j(GOOD)])
        args = parse(["validate", str(path), "--json"])
        if scenario == "badhint":
            args.format = "klingon"
        assert cli.cmd_validate(args) == expected[1]
        payload = last_json(capsys.readouterr().out)
        assert payload["error"] == expected[0] and payload["is_valid"] is False
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION

    def test_read_failure_payload(self, tmp_path, monkeypatch, capsys):
        p = _write(tmp_path, [_j(GOOD)])
        import builtins

        real_open = builtins.open

        def deny(path, *a, **k):
            if str(path) == str(p):
                raise PermissionError("denied")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", deny)
        assert cli.cmd_validate(parse(["validate", str(p), "--json"])) == cli.EXIT_USER_ERROR
        payload = last_json(capsys.readouterr().out)
        assert payload["error"] == "read_failed" and "denied" in payload["detail"]
