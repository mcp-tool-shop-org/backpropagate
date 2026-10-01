"""Coverage tests for ``cmd_data_report`` and ``cmd_data_split`` in cli.py.

Nothing is mocked: real JSONL files on ``tmp_path`` go through the real,
torch-free ``analyze_dataset`` / ``split_dataset``.
"""

from __future__ import annotations

import builtins
import json

import pytest

from backpropagate import cli
from tests.helpers.cli_cov_support import last_json, parse


def _words(i: int, tag: str) -> str:
    # Rows share no word shingles, so they are never near-duplicates of each other.
    return " ".join(f"{tag}{i}w{j}" for j in range(14))


def _convo(i: int, extra: str = "") -> str:
    return json.dumps({
        "messages": [
            {"role": "user", "content": _words(i, "q") + extra},
            {"role": "assistant", "content": _words(i, "a") + extra},
        ]
    })


def _write(tmp_path, lines, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def clean(tmp_path):
    return _write(tmp_path, [_convo(i) for i in range(12)])


@pytest.fixture
def duplicated(tmp_path):
    return _write(tmp_path, [_convo(1)] * 6 + [_convo(i) for i in range(2, 8)])


def _deny_open_for(monkeypatch, target):
    real_open = builtins.open

    def deny(path, *a, **k):
        if str(path) == str(target):
            raise PermissionError("denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", deny)


class TestDataReportInputs:
    def test_missing_dataset(self, tmp_path, capsys):
        assert cli.cmd_data_report(parse(["data", "report", str(tmp_path / "no.jsonl")])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Dataset not found" in captured.err and "no HuggingFace download" in captured.out

    def test_directory(self, tmp_path, capsys):
        assert cli.cmd_data_report(parse(["data", "report", str(tmp_path)])) == cli.EXIT_USER_ERROR
        assert "is a directory" in capsys.readouterr().err

    def test_against_missing(self, clean, tmp_path, capsys):
        args = parse(["data", "report", str(clean), "--against", str(tmp_path / "nope.jsonl")])
        assert cli.cmd_data_report(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "--against dataset not found" in captured.err and "train/test contamination" in captured.out

    def test_against_is_directory(self, clean, tmp_path, capsys):
        args = parse(["data", "report", str(clean), "--against", str(tmp_path)])
        assert cli.cmd_data_report(args) == cli.EXIT_USER_ERROR

    def test_not_utf8(self, tmp_path, capsys):
        p = tmp_path / "latin.jsonl"
        p.write_bytes(b'{"text": "caf\xe9"}\n')
        assert cli.cmd_data_report(parse(["data", "report", str(p)])) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "not valid UTF-8" in captured.err and "iconv" in captured.out

    def test_unreadable(self, clean, monkeypatch, capsys):
        _deny_open_for(monkeypatch, clean)
        assert cli.cmd_data_report(parse(["data", "report", str(clean)])) == cli.EXIT_USER_ERROR
        assert "Could not read dataset: denied" in capsys.readouterr().err

    def test_no_parseable_rows(self, tmp_path, capsys):
        p = _write(tmp_path, ["{bad", "", "nope"])
        assert cli.cmd_data_report(parse(["data", "report", str(p)])) == cli.EXIT_DATA_ERR
        assert "no parseable rows" in capsys.readouterr().err

    @pytest.mark.parametrize("scenario", ["missing", "dir", "against", "utf8", "read", "empty"])
    def test_json_error_payloads(self, tmp_path, clean, monkeypatch, capsys, scenario):
        expected_error = {
            "missing": "dataset_not_found", "dir": "dataset_path_is_directory",
            "against": "against_not_found", "utf8": "not_utf8", "read": "read_failed",
            "empty": "no_parseable_rows",
        }[scenario]
        expected_code = cli.EXIT_DATA_ERR if scenario == "empty" else cli.EXIT_USER_ERROR
        argv = ["data", "report", "--json"]
        if scenario == "missing":
            argv.insert(2, str(tmp_path / "no.jsonl"))
        elif scenario == "dir":
            argv.insert(2, str(tmp_path))
        elif scenario == "against":
            argv[2:2] = [str(clean), "--against", str(tmp_path / "nope.jsonl")]
        elif scenario == "utf8":
            p = tmp_path / "latin.jsonl"
            p.write_bytes(b'{"text": "caf\xe9"}\n')
            argv.insert(2, str(p))
        elif scenario == "read":
            _deny_open_for(monkeypatch, clean)
            argv.insert(2, str(clean))
        else:
            argv.insert(2, str(_write(tmp_path, ["{bad"])))
        assert cli.cmd_data_report(parse(argv)) == expected_code
        payload = last_json(capsys.readouterr().out)
        assert payload["error"] == expected_error
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION


class TestDataReportAnalysis:
    def test_clean_dataset_is_advisory_pass(self, clean, capsys):
        assert cli.cmd_data_report(parse(["data", "report", str(clean)])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Dataset quality: PASS" in out

    def test_json_payload_carries_report(self, clean, capsys):
        assert cli.cmd_data_report(parse(["data", "report", str(clean), "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION
        assert payload["verdict"] == "PASS"

    def test_duplicate_gate_trips_to_65(self, duplicated, capsys):
        args = parse(["data", "report", str(duplicated), "--fail-on-dups", "0.1"])
        assert cli.cmd_data_report(args) == cli.EXIT_DATA_ERR
        captured = capsys.readouterr()
        assert "Dataset quality: FAIL" in captured.err

    def test_duplicate_gate_trip_json(self, duplicated, capsys):
        args = parse(["data", "report", str(duplicated), "--fail-on-dups", "0.1", "--json"])
        assert cli.cmd_data_report(args) == cli.EXIT_DATA_ERR
        assert last_json(capsys.readouterr().out)["verdict"] == "FAIL"

    def test_gate_log_failure_does_not_change_exit(self, duplicated, monkeypatch):
        def boom(name):
            raise RuntimeError("log down")

        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        args = parse(["data", "report", str(duplicated), "--fail-on-dups", "0.1"])
        assert cli.cmd_data_report(args) == cli.EXIT_DATA_ERR

    def test_duplicates_without_gate_warn_only(self, duplicated, capsys):
        assert cli.cmd_data_report(parse(["data", "report", str(duplicated)])) == cli.EXIT_OK
        assert "Dataset quality: WARN" in capsys.readouterr().out

    def test_contamination_against_heldout(self, clean, tmp_path, capsys):
        heldout = _write(tmp_path, [_convo(0), _convo(1), _convo(99)], name="held.jsonl")
        args = parse(["data", "report", str(clean), "--against", str(heldout),
                      "--fail-on-contamination", "0.01", "--json"])
        assert cli.cmd_data_report(args) == cli.EXIT_DATA_ERR
        payload = last_json(capsys.readouterr().out)
        assert payload["verdict"] == "FAIL"

    def test_max_samples_caps_rows_read(self, clean, capsys):
        args = parse(["data", "report", str(clean), "--max-samples", "4", "--json"])
        assert cli.cmd_data_report(args) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["total_rows"] == 4 and payload["parseable_rows"] == 4


class TestDataSplit:
    def test_missing_and_directory(self, tmp_path, capsys):
        assert cli.cmd_data_split(parse(["data", "split", str(tmp_path / "no.jsonl")])) == cli.EXIT_USER_ERROR
        assert "Dataset not found" in capsys.readouterr().err
        assert cli.cmd_data_split(parse(["data", "split", str(tmp_path)])) == cli.EXIT_USER_ERROR
        assert "is a directory" in capsys.readouterr().err

    def test_not_utf8_and_unreadable(self, tmp_path, clean, monkeypatch, capsys):
        p = tmp_path / "latin.jsonl"
        p.write_bytes(b'{"text": "caf\xe9"}\n')
        assert cli.cmd_data_split(parse(["data", "split", str(p)])) == cli.EXIT_USER_ERROR
        assert "not valid UTF-8" in capsys.readouterr().err
        _deny_open_for(monkeypatch, clean)
        assert cli.cmd_data_split(parse(["data", "split", str(clean)])) == cli.EXIT_USER_ERROR
        assert "Could not read dataset: denied" in capsys.readouterr().err

    def test_no_parseable_rows(self, tmp_path, capsys):
        p = _write(tmp_path, ["{bad"])
        assert cli.cmd_data_split(parse(["data", "split", str(p)])) == cli.EXIT_USER_ERROR
        assert "nothing to split" in capsys.readouterr().err

    def test_default_outputs_next_to_input(self, clean, capsys):
        assert cli.cmd_data_split(parse(["data", "split", str(clean), "--heldout-ratio", "0.25", "--seed", "3"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        train = clean.parent / "d.train.jsonl"
        held = clean.parent / "d.heldout.jsonl"
        train_rows = train.read_text(encoding="utf-8").splitlines()
        held_rows = held.read_text(encoding="utf-8").splitlines()
        assert len(train_rows) == 9 and len(held_rows) == 3
        assert set(train_rows).isdisjoint(held_rows)
        assert "Dataset split complete" in out and "n_train: 9" in out and "(seed 3)" in out
        assert str(held) in out

    def test_split_is_deterministic_for_a_seed(self, clean, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        for d in (a, b):
            args = parse(["data", "split", str(clean), "--seed", "5",
                          "--out-train", str(d / "t.jsonl"), "--out-heldout", str(d / "h.jsonl")])
            assert cli.cmd_data_split(args) == cli.EXIT_OK
        assert (a / "h.jsonl").read_text(encoding="utf-8") == (b / "h.jsonl").read_text(encoding="utf-8")

    def test_bad_ratio(self, clean, capsys):
        args = parse(["data", "split", str(clean)])
        args.heldout_ratio = 1.5
        assert cli.cmd_data_split(args) == cli.EXIT_USER_ERROR
        assert "[CONFIG_INVALID_SETTING]" in capsys.readouterr().err

    def test_too_few_rows(self, tmp_path, capsys):
        p = _write(tmp_path, [_convo(1)])
        assert cli.cmd_data_split(parse(["data", "split", str(p)])) == cli.EXIT_USER_ERROR
        assert "[CONFIG_INVALID_SETTING]" in capsys.readouterr().err

    def test_unwritable_output(self, clean, tmp_path, capsys):
        blocker = tmp_path / "blocker"
        blocker.write_text("file", encoding="utf-8")
        args = parse(["data", "split", str(clean), "--out-train", str(blocker / "t.jsonl")])
        assert cli.cmd_data_split(args) == cli.EXIT_USER_ERROR
        captured = capsys.readouterr()
        assert "Could not write split output" in captured.err and "writable" in captured.out

    def test_log_failure_is_ignored(self, clean, monkeypatch, capsys):
        def boom(name):
            raise RuntimeError("log down")

        monkeypatch.setattr("backpropagate.logging_config.get_logger", boom)
        assert cli.cmd_data_split(parse(["data", "split", str(clean)])) == cli.EXIT_OK
