"""Unit tests for scripts/build_msix.py (MSIX packaging, 1.8.2 PR C).

Only the pure functions are tested here (version mapping, requirements
filtering, path/size gates, manifest rendering, embedded templates). The full
pipeline runs by hand on the build rig — it fetches multi-GB pinned artifacts
and needs the 5090 for the torch gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import tempfile
import zipfile
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

    def test_min_os_version_supports_uap10_parameters(self, mod):
        m = re.search(r'MinVersion="([\d.]+)"', mod.render_manifest("1.8.2.0"))
        assert m, "manifest must pin a MinVersion"
        parts = tuple(int(p) for p in m.group(1).split("."))
        # uap10:Parameters needs Windows 10 2004 (19041); on older builds it is
        # silently ignored and the Start tile would open a bare python REPL.
        assert parts >= (10, 0, 19041, 0)


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

    def test_launcher_cs_swallows_ctrlc_for_python_teardown(self, mod):
        src = mod._LAUNCHER_CS
        # Ctrl+C reaches python on the same console (its own teardown runs);
        # the launcher keeps waiting and returns python's exit code.
        assert "Console.CancelKeyPress" in src
        assert "e.Cancel = true" in src
        assert "return p.ExitCode" in src

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


class TestSideloadSign:
    """sideload_sign must destroy the signing key after use and print exact,
    thumbprint-bearing cleanup commands (the agent never runs trust-store
    commands itself - a human admin runs them)."""

    THUMB = "A1B2C3D4E5F60718293A4B5C6D7E8F9012345678"

    def _fake_run(self, calls, ok=True):
        def fake(cmd, **kwargs):
            calls.append(list(cmd))
            script = cmd[-1]
            stdout = ""
            if "New-SelfSignedCertificate" in script:
                if ok:
                    stdout = self.THUMB + "\n"
                pfx = re.search(r"-FilePath '([^']+\.pfx)'", script).group(1)
                cer = re.search(r"-FilePath '([^']+\.cer)'", script).group(1)
                Path(pfx).write_bytes(b"PRIVATE KEY MATERIAL")
                Path(cer).write_bytes(b"CERT")
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        return fake

    def test_key_destroyed_and_thumbprint_reported(self, mod, tmp_path, monkeypatch, capsys):
        calls: list[list[str]] = []
        monkeypatch.setattr(mod.shutil, "which", lambda name: "pwsh")
        monkeypatch.setattr(mod, "_find_sdk_tool", lambda name: Path("signtool.exe"))
        monkeypatch.setattr(mod, "_run", lambda cmd: calls.append(list(cmd)))
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run(calls))
        msix = tmp_path / "backpropagate_1.8.2.0_x64.msix"
        msix.write_bytes(b"msix")

        mod.sideload_sign(msix)

        cert_dir = tmp_path / "sideload-cert"
        assert not (cert_dir / "backpropagate-sideload.pfx").exists()  # key destroyed
        cer = cert_dir / "backpropagate-sideload.cer"
        assert cer.read_bytes() == b"CERT"  # public cert kept
        removals = [c for c in calls if "Remove-Item 'Cert:" in c[-1]]
        assert len(removals) == 1
        assert f"My\\{self.THUMB}" in removals[0][-1] and "-DeleteKey" in removals[0][-1]
        out = capsys.readouterr().out
        assert self.THUMB in out
        assert f"certutil -delstore TrustedPeople {self.THUMB}" in out
        assert f'Remove-Item "{cer}"' in out
        assert "private" in out and "deleted" in out

    def test_missing_thumbprint_refuses_and_still_destroys_the_key(
        self, mod, tmp_path, monkeypatch
    ):
        calls: list[list[str]] = []
        monkeypatch.setattr(mod.shutil, "which", lambda name: "pwsh")
        monkeypatch.setattr(mod, "_find_sdk_tool", lambda name: Path("signtool.exe"))
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run(calls, ok=False))
        msix = tmp_path / "x.msix"
        msix.write_bytes(b"msix")
        with pytest.raises(RuntimeError, match="thumbprint"):
            mod.sideload_sign(msix)

        assert not (tmp_path / "sideload-cert" / "backpropagate-sideload.pfx").exists()
        # No thumbprint to target: the script's own throwaway certs are
        # removed by their friendly name instead.
        sweeps = [c for c in calls if "Get-ChildItem Cert:" in c[-1]]
        assert len(sweeps) == 1
        assert "'backpropagate sideload test'" in sweeps[0][-1]
        assert "-DeleteKey" in sweeps[0][-1]

    def test_missing_signtool_creates_no_key(self, mod, tmp_path, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(mod.shutil, "which", lambda name: "pwsh")

        def no_sdk(name):
            raise RuntimeError("Windows SDK not found")

        monkeypatch.setattr(mod, "_find_sdk_tool", no_sdk)
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run(calls))
        msix = tmp_path / "x.msix"
        msix.write_bytes(b"msix")
        with pytest.raises(RuntimeError, match="SDK"):
            mod.sideload_sign(msix)

        assert not any("New-SelfSignedCertificate" in c[-1] for c in calls)
        assert not (tmp_path / "sideload-cert" / "backpropagate-sideload.pfx").exists()

    def test_sign_failure_still_destroys_the_key(self, mod, tmp_path, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(mod.shutil, "which", lambda name: "pwsh")
        monkeypatch.setattr(mod, "_find_sdk_tool", lambda name: Path("signtool.exe"))

        def boom(cmd):
            calls.append(list(cmd))
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(mod, "_run", boom)
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run(calls))
        msix = tmp_path / "backpropagate_1.8.2.0_x64.msix"
        msix.write_bytes(b"msix")

        with pytest.raises(subprocess.CalledProcessError):
            mod.sideload_sign(msix)

        cert_dir = tmp_path / "sideload-cert"
        assert not (cert_dir / "backpropagate-sideload.pfx").exists()  # key destroyed despite the failure
        removals = [c for c in calls if "Remove-Item 'Cert:" in c[-1]]
        assert len(removals) == 1 and self.THUMB in removals[0][-1]


def _zip_members(path: Path, tag: str, files: dict[str, bytes]) -> None:
    root = f"llama.cpp-{tag}/"
    with zipfile.ZipFile(path, "w") as zf:
        for rel, data in files.items():
            zf.writestr(root + rel, data)


class _Body:
    def __init__(self, data: bytes):
        self._data = data
        self._off = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._data) - self._off
        chunk = self._data[self._off:self._off + n]
        self._off += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestStoreEditionMarker:
    def test_marker_is_staged_next_to_the_interpreter(self, mod, tmp_path):
        python_home = tmp_path / "App" / "python"
        python_home.mkdir(parents=True)
        dest = mod.stage_store_edition_marker(python_home)
        assert dest == python_home / mod.STORE_EDITION_MARKER_NAME
        assert dest.read_text(encoding="utf-8") == mod.STORE_EDITION_MARKER_TEXT

    def test_sitecustomize_overwrites_the_opt_in_from_the_marker(self, mod):
        import backpropagate.config as cfg

        src = mod._SITECUSTOMIZE
        assert f'"{mod.STORE_EDITION_MARKER_NAME}"' in src
        assert mod.STORE_EDITION_MARKER_NAME == cfg.STORE_EDITION_MARKER_NAME
        assert '_os.environ["BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE"] = "false"' in src
        assert 'setdefault("BACKPROPAGATE_MODEL__TRUST_REMOTE_CODE"' not in src
        assert "Program Files" not in src


class TestLlamaCppManifest:
    """Every copied llama.cpp file is pinned, cached archive or not."""

    def _synthetic(self):
        files = {
            "LICENSE": b"license-body",
            "convert_hf_to_gguf.py": b"converter",
            "gguf-py/gguf/__init__.py": b"init",
        }
        manifest = {
            rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()
        }
        return files, manifest

    def test_matching_set_passes(self, mod):
        files, manifest = self._synthetic()
        mod.verify_llamacpp_manifest(files, manifest)

    def test_changed_file_fails(self, mod, tmp_path):
        files, manifest = self._synthetic()
        files = dict(files)
        files["LICENSE"] = b"tampered"
        archive = tmp_path / "changed.zip"
        _zip_members(archive, mod.LLAMACPP_TAG, files)
        with zipfile.ZipFile(archive) as zf:
            got = mod.llamacpp_archive_members(zf, mod.LLAMACPP_TAG)
        with pytest.raises(RuntimeError, match="changed: LICENSE"):
            mod.verify_llamacpp_manifest(got, manifest)

    def test_extra_gguf_file_fails_and_unrelated_files_do_not(self, mod, tmp_path):
        files, manifest = self._synthetic()
        archive = tmp_path / "extra.zip"
        _zip_members(
            archive,
            mod.LLAMACPP_TAG,
            {
                **files,
                "README.md": b"not copied",
                "gguf-py/gguf/extra.py": b"extra",
            },
        )
        with zipfile.ZipFile(archive) as zf:
            got = mod.llamacpp_archive_members(zf, mod.LLAMACPP_TAG)
        assert "README.md" not in got
        with pytest.raises(RuntimeError, match="not in manifest: gguf-py/gguf/extra.py"):
            mod.verify_llamacpp_manifest(got, manifest)

    def test_missing_file_fails(self, mod, tmp_path):
        files, manifest = self._synthetic()
        files = dict(files)
        del files["gguf-py/gguf/__init__.py"]
        archive = tmp_path / "missing.zip"
        _zip_members(archive, mod.LLAMACPP_TAG, files)
        with zipfile.ZipFile(archive) as zf:
            got = mod.llamacpp_archive_members(zf, mod.LLAMACPP_TAG)
        with pytest.raises(RuntimeError, match="missing: gguf-py/gguf/__init__.py"):
            mod.verify_llamacpp_manifest(got, manifest)

    def test_unsafe_member_is_refused(self, mod, tmp_path):
        archive = tmp_path / "unsafe.zip"
        _zip_members(archive, mod.LLAMACPP_TAG, {"gguf-py/gguf/../../LICENSE": b"x"})
        with zipfile.ZipFile(archive) as zf:
            with pytest.raises(RuntimeError, match="unsafe"):
                mod.llamacpp_archive_members(zf, mod.LLAMACPP_TAG)

    def test_cached_archive_is_checked_without_network(self, mod, tmp_path, monkeypatch):
        downloads = tmp_path / "downloads"
        downloads.mkdir()
        stage = tmp_path / "stage"
        files, _manifest = self._synthetic()
        files = dict(files)
        files["LICENSE"] = b"not-the-pin"
        archive = downloads / f"llama.cpp-{mod.LLAMACPP_TAG}.zip"
        _zip_members(archive, mod.LLAMACPP_TAG, files)

        def boom(*_a, **_k):
            raise AssertionError("cached archive must not be fetched again")

        monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
        with pytest.raises(RuntimeError, match="manifest mismatch"):
            mod.stage_llamacpp(downloads, stage)

    def test_manifest_text_is_sorted_path_sha_lines(self, mod, tmp_path):
        files, manifest = self._synthetic()
        archive = tmp_path / "ok.zip"
        _zip_members(archive, "b99999", files)
        text = mod.llamacpp_manifest_text(archive, "b99999", commit="abc")
        lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
        assert lines == sorted(lines)
        assert lines[0].startswith("LICENSE ")
        assert manifest["LICENSE"] in lines[0]
        assert text.startswith("# tag b99999 commit abc\n")

    def test_print_flag_writes_the_manifest_and_does_not_need_out(self, mod, monkeypatch, capsys):
        monkeypatch.setattr(
            mod,
            "render_llamacpp_manifest_for_tag",
            lambda tag: f"# tag {tag} commit x\nLICENSE abc\n",
        )
        assert mod.main(["--print-llamacpp-manifest"]) == 0
        assert "LICENSE abc" in capsys.readouterr().out
        assert mod.main(["--print-llamacpp-manifest", "--llamacpp-tag", "b99999"]) == 0
        assert "b99999" in capsys.readouterr().out

    @pytest.mark.integration  # reaches github.com; run by hand, never in CI
    def test_manifest_constant_matches_pinned_tag(self, mod):
        """The constant was computed from the b11323 tag archive.

        See the comment on LLAMACPP_MANIFEST. A cache hit in the system temp
        directory is reused; a miss downloads the tag (commit-checked) once.
        Marked integration: unit tests stay off the network.
        """
        cache = Path(tempfile.gettempdir()) / f"llama.cpp-{mod.LLAMACPP_TAG}.zip"
        if not cache.is_file():
            mod.fetch_llamacpp_archive(
                cache, mod.LLAMACPP_TAG, commit=mod.LLAMACPP_COMMIT
            )
        with zipfile.ZipFile(cache) as zf:
            files = mod.llamacpp_archive_members(zf, mod.LLAMACPP_TAG)
        mod.verify_llamacpp_manifest(files, mod.LLAMACPP_MANIFEST)
        assert "LICENSE" in mod.LLAMACPP_MANIFEST
        assert "convert_hf_to_gguf.py" in mod.LLAMACPP_MANIFEST
        assert "gguf-py/gguf/__init__.py" in mod.LLAMACPP_MANIFEST


class TestNoticePins:
    def test_changed_body_fails(self, mod, tmp_path, monkeypatch):
        digest = hashlib.sha256(b"correct license\n").hexdigest()
        monkeypatch.setattr(
            mod,
            "NOTICES",
            [("Example (MIT)", "https://example.invalid/LICENSE", digest)],
        )
        monkeypatch.setattr(
            mod.urllib.request, "urlopen", lambda *_a, **_k: _Body(b"tampered license\n")
        )
        stage = tmp_path / "stage"
        stage.mkdir()
        with pytest.raises(RuntimeError, match="SHA-256"):
            mod.stage_notices(tmp_path, stage, tmp_path / "downloads")
        assert not (stage / "THIRD_PARTY_NOTICES.txt").exists()

    def test_matching_body_and_local_license_are_written(self, mod, tmp_path, monkeypatch):
        body = b"the license text\n"
        digest = hashlib.sha256(body).hexdigest()
        monkeypatch.setattr(
            mod,
            "NOTICES",
            [
                ("backpropagate (MIT)", None, None),
                ("Example (MIT)", "https://example.invalid/LICENSE", digest),
            ],
        )
        monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_k: _Body(body))
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "LICENSE").write_text("our license", encoding="utf-8")
        stage = tmp_path / "stage"
        stage.mkdir()
        mod.stage_notices(repo, stage, tmp_path / "downloads")
        text = (stage / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
        assert "our license" in text
        assert "the license text" in text
