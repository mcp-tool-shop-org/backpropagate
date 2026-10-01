"""Coverage tests for ``cmd_info`` and its introspection helpers in cli.py.

Real: the argparse wiring, settings introspection, JSON / table rendering.
Mocked (real boundaries): the GPU probes (``get_gpu_info`` / ``get_gpu_status``,
which talk to CUDA / NVML), package-metadata lookups, and ``shutil.which`` for
the cloudflared probe.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from backpropagate import cli
from tests.helpers.cli_cov_support import last_json, parse


class TestDetectInstalledVersions:
    def test_missing_unknown_and_present(self, monkeypatch):
        from importlib import metadata

        def fake_version(name):
            if name == "torch":
                raise metadata.PackageNotFoundError(name)
            if name == "wandb":
                raise RuntimeError("corrupt metadata")
            return "9.9.9"

        monkeypatch.setattr(metadata, "version", fake_version)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")
        out = cli._detect_installed_versions()
        assert out["torch"] == "not installed"
        assert out["wandb"] == "unknown"
        assert out["transformers"] == "9.9.9"
        assert out["cloudflared"] == "installed"

    def test_cloudflared_absent_and_probe_failure(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert cli._detect_installed_versions()["cloudflared"] == "not installed"

        def boom(name):
            raise OSError("PATH unreadable")

        monkeypatch.setattr("shutil.which", boom)
        assert cli._detect_installed_versions()["cloudflared"] == "unknown"


class TestEnumerateEnvVars:
    def test_real_settings_rows_sorted_and_include_extras(self):
        rows = cli._enumerate_env_vars()
        names = [r["env_var"] for r in rows]
        assert names == sorted(names)
        assert len(set(names)) == len(names)
        assert "BACKPROPAGATE_LOG_LEVEL" in names
        assert "BACKPROPAGATE_QUIET_TOKEN_HINT" in names
        assert all(set(r) == {"env_var", "default", "type", "description"} for r in rows)

    def test_fallback_without_pydantic_settings(self, monkeypatch):
        monkeypatch.setattr("backpropagate.config.PYDANTIC_SETTINGS_AVAILABLE", False)
        rows = cli._enumerate_env_vars()
        names = {r["env_var"] for r in rows}
        assert "BACKPROPAGATE_DEFER_FEATURE_DETECTION" in names
        # The hand-curated row is not duplicated by the logging list.
        assert [r["env_var"] for r in rows].count("BACKPROPAGATE_DEFER_FEATURE_DETECTION") == 1
        assert "BACKPROPAGATE_LOG_LEVEL" in names

    def test_top_level_scalar_and_secret_fields(self, monkeypatch):
        class Leaf:
            model_config = {"env_prefix": "BACKPROPAGATE_LEAF__"}
            model_fields = {
                "plain": SimpleNamespace(default=3, annotation=int, description="a plain knob",
                                         json_schema_extra=None),
                "token": SimpleNamespace(default="hunter2", annotation=str, description="secret",
                                         json_schema_extra={"secret": True}),
            }

        class FakeSettings:
            model_fields = {
                "name": SimpleNamespace(default="x", annotation=str, description=" top-level ",
                                        json_schema_extra=None),
                "leaf": SimpleNamespace(default=None, annotation=Leaf, description="",
                                        json_schema_extra=None),
            }

        monkeypatch.setattr("backpropagate.config.Settings", FakeSettings)
        rows = {r["env_var"]: r for r in cli._enumerate_env_vars()}
        assert rows["BACKPROPAGATE_NAME"]["default"] == "x"
        assert rows["BACKPROPAGATE_NAME"]["description"] == "top-level"
        assert rows["BACKPROPAGATE_LEAF__PLAIN"]["default"] == "3"
        assert rows["BACKPROPAGATE_LEAF__PLAIN"]["type"] == "int"
        assert rows["BACKPROPAGATE_LEAF__TOKEN"]["default"] == "<secret>"
        assert "hunter2" not in json.dumps(list(rows.values()))

    def test_sub_config_without_env_prefix_uses_field_name(self, monkeypatch):
        class Leaf:
            model_config = {}
            model_fields = {"depth": SimpleNamespace(default=1, annotation=int, description="d",
                                                     json_schema_extra=None)}

        class FakeSettings:
            model_fields = {"inner": SimpleNamespace(default=None, annotation=Leaf, description="",
                                                     json_schema_extra=None)}

        monkeypatch.setattr("backpropagate.config.Settings", FakeSettings)
        names = {r["env_var"] for r in cli._enumerate_env_vars()}
        assert "BACKPROPAGATE_INNER__DEPTH" in names


class TestSafeDefaultAndTypeName:
    def test_defaults(self):
        from pydantic_core import PydanticUndefined

        assert cli._safe_default(SimpleNamespace(default=None)) == "None"
        assert cli._safe_default(SimpleNamespace(default=True)) == "true"
        assert cli._safe_default(SimpleNamespace(default=False)) == "false"
        assert cli._safe_default(SimpleNamespace(default=[1, 2])) == "[1, 2]"
        assert cli._safe_default(SimpleNamespace(default=(1,))) == "(1,)"
        assert cli._safe_default(SimpleNamespace(default={"a": 1})) == "{'a': 1}"
        assert cli._safe_default(SimpleNamespace(default=0.5)) == "0.5"
        assert cli._safe_default(SimpleNamespace(default=PydanticUndefined, default_factory=list)) == "<factory>"
        assert cli._safe_default(SimpleNamespace(default=PydanticUndefined, default_factory=None)) == ""
        assert cli._safe_default(SimpleNamespace(default=Ellipsis)) == ""

    def test_without_pydantic_core(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pydantic_core", None)  # makes the import raise ImportError
        assert cli._safe_default(SimpleNamespace()) == ""
        assert cli._safe_default(SimpleNamespace(default=7)) == "7"

    def test_type_name(self):
        assert cli._type_name(None) == "any"
        assert cli._type_name(int) == "int"
        assert cli._type_name(str | None) == "str | None"


class TestInfoSubcommandTiers:
    def test_table(self, capsys):
        assert cli.cmd_info(parse(["info", "--subcommand-tiers"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "SUBCOMMAND" in out and "TIER" in out
        assert "stable — documented contract" in out
        assert "deprecated-prefer-X" in out
        for name in cli.SUBCOMMAND_TIERS:
            assert name in out

    def test_json(self, capsys):
        assert cli.cmd_info(parse(["info", "--subcommand-tiers", "--json"])) == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION
        assert payload["subcommand_tiers"] == dict(cli.SUBCOMMAND_TIERS)

    def test_empty_registry(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "SUBCOMMAND_TIERS", {})
        assert cli.cmd_info(parse(["info", "--subcommand-tiers"])) == cli.EXIT_OK
        assert "(no entries)" in capsys.readouterr().out

    def test_unknown_tier_sorts_last(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "SUBCOMMAND_TIERS", {"zeta": "stable", "alpha": "mystery", "beta": "experimental"})
        assert cli.cmd_info(parse(["info", "--subcommand-tiers"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert out.index("zeta") < out.index("beta") < out.index("alpha")


class TestInfoEnvVars:
    def test_table(self, capsys):
        assert cli.cmd_info(parse(["info", "--env-vars"])) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "ENV_VAR" in out and "DEFAULT" in out and "DESCRIPTION" in out
        assert "BACKPROPAGATE_LOG_LEVEL" in out
        assert "environment variable(s)" in out

    def test_empty(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_enumerate_env_vars", lambda: [])
        assert cli.cmd_info(parse(["info", "--env-vars"])) == cli.EXIT_OK
        assert "No environment variables enumerated" in capsys.readouterr().out

    def test_json_rows(self, capsys):
        assert cli.cmd_info(parse(["info", "--env-vars", "--json"])) == cli.EXIT_OK
        rows = json.loads(capsys.readouterr().out)
        assert rows and all(r["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION for r in rows)


class TestInfoGpuAndFeatures:
    @pytest.fixture
    def run_info(self, monkeypatch, capsys):
        def _run(gpu, status=None, status_exc=None, features=None):
            monkeypatch.setattr("backpropagate.feature_flags.get_gpu_info", lambda: gpu)
            if status_exc is not None:
                def boom():
                    raise status_exc
                monkeypatch.setattr("backpropagate.gpu_safety.get_gpu_status", boom)
            else:
                monkeypatch.setattr("backpropagate.gpu_safety.get_gpu_status", lambda: status)
            if features is not None:
                for k, v in features.items():
                    monkeypatch.setitem(__import__("backpropagate.feature_flags", fromlist=["x"]).FEATURES, k, v)
            assert cli.cmd_info(parse(["info"])) == cli.EXIT_OK
            return capsys.readouterr().out

        return _run

    def test_memory_total_bytes_and_hot_temperature(self, run_info):
        out = run_info(
            {"available": True, "device_name": "RTX Test", "device_count": 2,
             "memory_total": 32 * 1024 ** 3, "vram_free_gb": 20.0},
            status=SimpleNamespace(temperature_c=90),
        )
        assert "RTX Test" in out and "Device count:" in out
        assert "32.0 GB" in out
        assert "VRAM Free:" in out and "20.0 GB" in out
        assert "90C" in out

    def test_legacy_field_names_and_warm_temperature(self, run_info):
        out = run_info(
            {"available": True, "name": "Legacy GPU", "vram_total_gb": 16.0},
            status=SimpleNamespace(temperature_c=75),
        )
        assert "Legacy GPU" in out
        assert "16.0 GB" in out
        assert "75C" in out

    def test_cool_temperature_and_no_vram_keys(self, run_info):
        out = run_info({"available": True, "device_name": "G"}, status=SimpleNamespace(temperature_c=40))
        assert "40C" in out
        assert "VRAM:" not in out

    def test_no_status_returned(self, run_info):
        out = run_info({"available": True, "device_name": "G"}, status=None)
        assert "Temperature" not in out

    def test_nvml_missing_is_tolerated(self, run_info):
        out = run_info({"available": True, "device_name": "G"}, status_exc=ImportError("pynvml"))
        assert "Temperature" not in out and "Features" in out

    def test_temperature_probe_error_is_tolerated(self, run_info):
        out = run_info({"available": True, "device_name": "G"}, status_exc=RuntimeError("nvml dead"))
        assert "Temperature" not in out and "Features" in out

    def test_no_gpu_branch(self, run_info):
        out = run_info({"available": False})
        assert "No GPU detected" in out

    def test_missing_feature_shows_install_hint(self, run_info, monkeypatch):
        from backpropagate import feature_flags

        monkeypatch.setitem(feature_flags.INSTALL_HINTS, "zz_feature", "pip install zz")
        out = run_info({"available": False}, features={"zz_feature": False, "yy_feature": False})
        assert "[-]" in out and "zz_feature" in out
        assert "install with:" in out and "pip install zz" in out
        assert "yy_feature" in out  # no hint registered -> no 'install with' line for it


class TestInfoJsonLogging:
    def _logging_block(self, capsys):
        assert cli.cmd_info(parse(["info", "--json"])) == cli.EXIT_OK
        payload = last_json(capsys.readouterr().out)
        assert payload["schema_version"] == cli.CLI_JSON_SCHEMA_VERSION
        return payload["logging"]

    @pytest.mark.parametrize(
        "json_env, expected_format",
        [("true", "json"), ("false", "console")],
    )
    def test_explicit_log_json_env(self, monkeypatch, capsys, json_env, expected_format):
        monkeypatch.setenv("BACKPROPAGATE_LOG_JSON", json_env)
        monkeypatch.setenv("BACKPROPAGATE_LOG_LEVEL", "debug")
        monkeypatch.setenv("BACKPROPAGATE_LOG_FILE", "x.log")
        block = self._logging_block(capsys)
        assert block == {"level": "DEBUG", "format": expected_format, "file": "x.log", "json_var_set": True}

    @pytest.mark.parametrize("is_tty, expected", [(True, "console"), (False, "json")])
    def test_auto_detect_from_stderr_tty(self, monkeypatch, capsys, is_tty, expected):
        monkeypatch.delenv("BACKPROPAGATE_LOG_JSON", raising=False)
        monkeypatch.delenv("BACKPROPAGATE_LOG_FILE", raising=False)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: is_tty, raising=False)
        block = self._logging_block(capsys)
        assert block["format"] == expected
        assert block["file"] is None
        assert block["json_var_set"] is False
