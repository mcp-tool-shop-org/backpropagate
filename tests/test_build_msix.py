"""Unit tests for scripts/build_msix.py (MSIX packaging, 1.8.2 PR C).

Only the pure functions are tested here (version mapping, requirements
filtering, path/size gates, manifest rendering, embedded templates). The full
pipeline runs by hand on the build rig — it fetches multi-GB pinned artifacts
and needs the 5090 for the torch gate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_msix", REPO_ROOT / "scripts" / "build_msix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


class TestMsixVersion:
    def test_plain_xyz_maps_to_xyz0(self, mod):
        assert mod.msix_version("1.8.2") == "1.8.2.0"
        assert mod.msix_version("1.8.10") == "1.8.10.0"

    @pytest.mark.parametrize(
        "bad", ["1.8.2.1", "1.8.2rc2", "1.8", "1.8.2b1", "0.9.0", "1.8.x"]
    )
    def test_refuses_non_plain_versions(self, mod, bad):
        with pytest.raises(ValueError, match="refusing"):
            mod.msix_version(bad)


class TestFilterRequirements:
    SAMPLE = """# header
accelerate==1.15.0 \\
    --hash=sha256:aaaa
    # via backpropagate
nvidia-cublas==13.1.1.3 ; sys_platform == 'linux' \\
    --hash=sha256:bbbb
    # via torch
torch==2.12.1 \\
    --hash=sha256:cccc \\
    --hash=sha256:dddd
    # via backpropagate, trl
trl==0.24.0 \\
    --hash=sha256:eeee
    # via backpropagate
"""

    def test_torch_stanza_removed_with_continuations(self, mod):
        out = mod.filter_requirements(self.SAMPLE)
        assert "torch==2.12.1" not in out
        assert "cccc" not in out and "dddd" not in out
        assert "# via backpropagate, trl" not in out

    def test_neighbors_and_linux_marked_nvidia_survive(self, mod):
        out = mod.filter_requirements(self.SAMPLE)
        assert "accelerate==1.15.0" in out
        assert "trl==0.24.0" in out
        # nvidia lines carry sys_platform=='linux' markers: never installed by
        # uv on Windows, so they stay (keeps the export hash-complete).
        assert "nvidia-cublas==13.1.1.3" in out
        assert "# header" in out


class TestWindowsAppsPrefix:
    def test_prefix_shape(self, mod):
        prefix = mod.windowsapps_prefix("1.8.2.0")
        assert prefix.startswith("C:\\Program Files\\WindowsApps\\")
        assert "mcp-tool-shop.backpropagate_1.8.2.0_x64__" in prefix
        assert prefix.endswith("yn6b8xqrexa5j")


class TestMaxPathGate:
    def test_short_tree_passes(self, mod, tmp_path):
        (tmp_path / "App" / "python").mkdir(parents=True)
        (tmp_path / "App" / "python" / "python.exe").write_bytes(b"x")
        deepest, total = mod.check_max_path(tmp_path, "1.8.2.0")
        assert deepest == len("App\\python\\python.exe")
        assert total == len(mod.windowsapps_prefix("1.8.2.0")) + 1 + deepest

    def test_over_budget_refuses(self, mod, tmp_path, monkeypatch):
        deep = tmp_path / ("x" * 120)  # short enough to CREATE under pytest's deep %TEMP% (MAX_PATH on this rig), long enough for the monkeypatched limit below
        deep.write_bytes(b"x")
        monkeypatch.setattr(mod, "MAX_PATH_LIMIT", 100)
        with pytest.raises(RuntimeError, match="MAX_PATH"):
            mod.check_max_path(tmp_path, "1.8.2.0")


class TestSizeGate:
    def test_under_cap(self, mod):
        mod.check_size(1024, cap=2048)

    def test_over_cap_refuses(self, mod):
        with pytest.raises(RuntimeError, match="cap"):
            mod.check_size(4097, cap=2048)


class TestManifest:
    def test_identity_and_version(self, mod):
        text = mod.render_manifest("1.8.2.0")
        assert 'Name="mcp-tool-shop.backpropagate"' in text
        assert "CN=5305D976-6952-4F00-9C21-3A5DB090359F" in text
        assert 'Version="1.8.2.0"' in text
        assert "<PublisherDisplayName>mcp-tool-shop</PublisherDisplayName>" in text

    def test_capabilities_and_entrypoints(self, mod):
        text = mod.render_manifest("1.8.2.0")
        assert 'rescap:Capability Name="runFullTrust"' in text
        # tile: raw python with parameters (console + banner + browser)
        assert 'Executable="App\\python\\python.exe"' in text
        assert 'uap10:Parameters="-m backpropagate ui --open-browser"' in text
        # one extension, two aliases, one csc-built launcher exe
        assert 'Executable="App\\backprop-launcher.exe"' in text
        assert 'Alias="backprop.exe"' in text
        assert 'Alias="backpropagate.exe"' in text

    def test_well_formed_xml(self, mod):
        import xml.etree.ElementTree as ET

        ET.fromstring(mod.render_manifest("1.8.2.0"))  # raises on malformed


class TestEmbeddedTemplates:
    def test_pth_enables_site(self, mod):
        assert "Lib\\site-packages" in mod._PTH
        assert "import site" in mod._PTH
        assert "python312.zip" in mod._PTH

    def test_sitecustomize_derives_paths_without_absolutes(self, mod):
        src = mod._SITECUSTOMIZE
        assert "BACKPROPAGATE_LLAMA_CPP_PATH" in src
        assert "BACKPROPAGATE_UI_PAYLOAD_DIR" in src
        assert "Program Files" not in src  # never bake the install prefix

    def test_launcher_cs_quotes_and_no_baked_paths(self, mod):
        src = mod._LAUNCHER_CS
        assert "WindowsApps" not in src
        assert 'Path.Combine(dir, "python", "python.exe")' in src
        assert "-m backpropagate" in src
        assert "Quote(" in src

    def test_torch_gate_requires_cuda_13_and_a_real_op(self, mod):
        script = mod.torch_gate_script()
        assert "(13, 0)" in script
        assert "is_available" in script
        assert "device='cuda'" in script


class TestPins:
    def test_torch_pin_is_not_pypi(self, mod):
        assert "pytorch.org" in mod.TORCH_WHEEL_URL
        assert mod.TORCH_WHEEL_SHA256
        assert mod.MIN_NVIDIA_DRIVER == 580
        assert mod.TORCH_CU_VARIANT == "cu130"

    def test_llamacpp_pin(self, mod):
        assert mod.LLAMACPP_TAG == "b11323"
        assert len(mod.LLAMACPP_COMMIT) == 40

    def test_size_cap_is_20_gib(self, mod):
        assert mod.SIZE_CAP_BYTES == 20 * 1024**3


class TestPayloadProducerBudget:
    """The producer's MAX_PATH model: reflex prod rebuilds the frontend from
    node_modules at EVERY launch (setup_frontend_prod → build(), reflex 0.9.5),
    so node_modules ships and the workdir depth budget must hold it."""

    @pytest.fixture(scope="class")
    def prod(self):
        spec = importlib.util.spec_from_file_location(
            "build_ui_frontend", REPO_ROOT / "scripts" / "build_ui_frontend.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_budget_matches_model(self, prod):
        # 259 - (15 LOCALAPPDATA-base + 24 user + 16 product\ui + 20 version dir + 6 \.web\)
        assert prod.web_depth_budget() == 178

    def test_model_constants_reasonable(self, prod):
        assert 20 <= prod._U_MAX <= 30
        assert prod.RUNTIME_PREFIX_MODEL == 81
