#!/usr/bin/env python3
"""Build the Microsoft Store MSIX package for backpropagate (1.8.2 track).

Pipeline (each step prints progress; everything under ``--out``):

1. Stage the layout: ``App/python`` (embedded CPython 3.12.10, hash-pinned),
   ``App/python/Lib/site-packages`` (the lock's ``--extra ui`` closure with
   PyPI's CPU-only torch REMOVED, replaced by the pinned cu130 wheel whose
   CUDA runtime is vendored — verified end-to-end with a real CUDA op),
   the UI frontend payload (web.zip + pinned bun, via build_ui_frontend.py),
   ``App/vendor/llama.cpp`` (convert_hf_to_gguf.py + gguf-py, one pinned tag),
   ``App/backprop-launcher.exe`` (csc-built), generated tile/Store logos from
   assets/logo.png, ``THIRD_PARTY_NOTICES.txt``.
2. Gates that refuse to pack: pyproject version must be X.Y.Z (-> X.Y.Z.0,
   4th part is reserved for the Store); torch must report CUDA >= 13.0 and
   run an op on the build GPU; no staged path may exceed MAX_PATH under the
   real WindowsApps install prefix (LongPathsEnabled=0 machines); the staged
   tree must be <= 20 GiB.
3. ``makeappx.exe pack`` -> unsigned .msix (the Store re-signs after
   certification). Self-signing happens ONLY with ``--sideload-test`` (local
   verification), which creates a throwaway cert whose subject matches the
   Partner Center publisher and prints the trust/import instructions.

Requirements on the build machine: Windows, the project venv Python with
Pillow, ``uv`` on PATH, the Windows SDK (makeappx/signtool), ~30 GB scratch.
Network: python.org, download.pytorch.org, github.com (bun + llama.cpp +
licenses), npmjs (the payload's production build).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Pinned inputs. Bump in lockstep with uv.lock / the handoff doc.

PYTHON_VERSION = "3.12.10"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
PYTHON_EMBED_SHA256 = "4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3"

# PyPI's win_amd64 torch wheel for 2.12.1 is CPU-only (122 MB). The CUDA build
# for Windows lives on download.pytorch.org; cu128 tops out at 2.9.1 there, so
# the minimal >=12.8 variant carrying 2.12.1 is cu130 (CUDA 13.0). The cu130
# win wheel VENDORS the CUDA runtime (no nvidia-* Requires-Dist) — the whole
# GPU stack arrives in this one file.
TORCH_VERSION = "2.12.1"
TORCH_CU_VARIANT = "cu130"
TORCH_WHEEL_URL = (
    "https://download-r2.pytorch.org/whl/cu130/"
    "torch-2.12.1%2Bcu130-cp312-cp312-win_amd64.whl"
)
TORCH_WHEEL_SHA256 = "52c5da6a0898d5d3473c02bd304b7a3bc0b72e351c6f3bfa0783e45ef9f4cd61"
MIN_NVIDIA_DRIVER = 580  # CUDA 13.x runtime requirement; older drivers fall back to CPU silently

LLAMACPP_TAG = "b11323"
LLAMACPP_COMMIT = "f11d642a27b921cf22b6a8beb1b899f960fedcde"
LLAMACPP_ARCHIVE_URL = (
    f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMACPP_TAG}.zip"
)
LLAMACPP_CONVERTER_SHA256 = "e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f"
LLAMACPP_GGUF_INIT_SHA256 = "3ccfc0104cd7ea88c6028743b7bf3f2c89b5f474425de03a217a6072320d7c2f"

# Package identity (Partner Center product 9MVXLZVL3TMT). Values are
# case-sensitive; re-read the Publisher on the Product identity page before a
# Store build:
# https://partner.microsoft.com/dashboard/products/9MVXLZVL3TMT/identity
IDENTITY_NAME = "mcp-tool-shop.backpropagate"
IDENTITY_PUBLISHER = "CN=5305D976-6952-4F00-9C21-3A5DB090359F"
PUBLISHER_DISPLAY_NAME = "mcp-tool-shop"
PUBLISHER_ID_SUFFIX = "yn6b8xqrexa5j"
DISPLAY_NAME = "backpropagate"

SIZE_CAP_BYTES = 20 * 1024**3  # 20 GiB; the Store cap is 25 GB (decimal)
MAX_PATH_LIMIT = 259  # usable chars under MAX_PATH=260 (excluding NUL)

NOTICES = [
    (
        "backpropagate (MIT)",
        None,  # local: <repo>/LICENSE
    ),
    (
        "CPython (PSF License)",
        "https://raw.githubusercontent.com/python/cpython/v3.12.10/LICENSE",
    ),
    (
        "PyTorch (BSD-3-Clause)",
        "https://raw.githubusercontent.com/pytorch/pytorch/v2.12.1/LICENSE",
    ),
    (
        "bun (MIT)",
        "https://raw.githubusercontent.com/oven-sh/bun/bun-v1.3.13/LICENSE.md",
    ),
    (
        "Reflex (MIT)",
        "https://raw.githubusercontent.com/reflex-dev/reflex/v0.9.5.post2/LICENSE",
    ),
    (
        "llama.cpp (MIT)",
        None,  # taken from the extracted archive (shipped in App/vendor too)
    ),
]

# --------------------------------------------------------------------------

_LAUNCHER_CS = r"""// backprop-launcher.exe — executes the packaged CPython with the user's args.
// Exists because App Execution Alias extensions cannot bake arguments and a
// packaged python.exe is not directly alias-able; one exe serves both the
// `backprop` and `backpropagate` aliases (identical semantics).
// Build with the OS framework compiler: %WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

class BackpropLauncher {
    static int Main(string[] args) {
        var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        var python = Path.Combine(dir, "python", "python.exe");
        if (!File.Exists(python)) {
            Console.Error.WriteLine("backpropagate: packaged python not found at " + python);
            return 1;
        }
        // Ctrl+C is delivered to every process attached to this console. Swallow
        // it here so the launcher survives and keeps waiting on python, whose own
        // Ctrl+C teardown runs; the shell prompt then returns with python's real
        // exit code instead of reappearing mid-shutdown.
        Console.CancelKeyPress += (s, e) => { e.Cancel = true; };
        var sb = new StringBuilder("-m backpropagate");
        foreach (var a in args) { sb.Append(' '); sb.Append(Quote(a)); }
        var psi = new ProcessStartInfo {
            FileName = python,
            Arguments = sb.ToString(),
            UseShellExecute = false,
            WorkingDirectory = Environment.CurrentDirectory,
        };
        try {
            using (var p = Process.Start(psi)) { p.WaitForExit(); return p.ExitCode; }
        } catch (Exception ex) {
            Console.Error.WriteLine("backpropagate: failed to start " + python + ": " + ex.Message);
            return 1;
        }
    }

    // CommandLineToArgvW-compatible quoting: backslashes are literal except
    // in runs that end at a quote or the closing quote (where they double);
    // embedded quotes are backslash-escaped. An empty/whitespace/quote-bearing
    // argument is wrapped in quotes.
    static string Quote(string s) {
        if (s.Length > 0 && s.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0) return s;
        var sb = new StringBuilder("\"");
        int i = 0;
        while (i < s.Length) {
            int slashes = 0;
            while (i < s.Length && s[i] == '\\') { slashes++; i++; }
            if (i == s.Length) { sb.Append('\\', slashes * 2); break; }
            if (s[i] == '"') { sb.Append('\\', slashes * 2 + 1); sb.Append('"'); i++; }
            else { sb.Append('\\', slashes); sb.Append(s[i]); i++; }
        }
        sb.Append('"');
        return sb.ToString();
    }
}
"""

_PTH = "python312.zip\n.\nLib\\site-packages\nimport site\n"

_SITECUSTOMIZE = '''"""backpropagate MSIX site bootstrap (runs at interpreter start via site).

Everything is derived from THIS python's location — no absolute paths are
baked in, because the WindowsApps install prefix changes with every version.
"""

import os as _os
from pathlib import Path as _Path

# .../App/python/Lib/site-packages/sitecustomize.py -> App/
_app_root = _Path(__file__).resolve().parents[3]

_llama = _app_root / "vendor" / "llama.cpp"
if (_llama / "convert_hf_to_gguf.py").is_file():
    _os.environ.setdefault("BACKPROPAGATE_LLAMA_CPP_PATH", str(_llama))

_payload = _app_root / "ui_frontend_payload"
if (_payload / "payload.json").is_file():
    _os.environ.setdefault("BACKPROPAGATE_UI_PAYLOAD_DIR", str(_payload))
'''

_MANIFEST_TEMPLATE = r"""<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5"
  xmlns:uap10="http://schemas.microsoft.com/appx/manifest/uap/windows10/10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap uap5 uap10 rescap">
  <Identity Name="{identity_name}" Publisher="{publisher}" Version="{msix_version}" ProcessorArchitecture="x64"/>
  <Properties>
    <DisplayName>{display_name}</DisplayName>
    <PublisherDisplayName>{publisher_display}</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>
  <Resources><Resource Language="en-us"/></Resources>
  <Dependencies>
    <!-- uap10:Parameters needs Windows 10 2004 (19041); on older builds it is
         silently ignored and the tile would open a bare python REPL -->
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0"/>
  </Dependencies>
  <Capabilities>
    <rescap:Capability Name="runFullTrust"/>
  </Capabilities>
  <Applications>
    <Application Id="BackpropagateUI" Executable="App\python\python.exe" EntryPoint="Windows.FullTrustApplication" uap10:Parameters="-m backpropagate ui --open-browser">
      <uap:VisualElements DisplayName="{display_name}" Description="Fine-tune LLMs on your GPU; export to GGUF/Ollama." Square150x150Logo="Assets\Square150x150Logo.png" Square44x44Logo="Assets\Square44x44Logo.png" BackgroundColor="transparent"/>
      <Extensions>
        <uap5:Extension Category="windows.appExecutionAlias" Executable="App\backprop-launcher.exe" EntryPoint="Windows.FullTrustApplication">
          <uap5:AppExecutionAlias>
            <uap5:ExecutionAlias Alias="backprop.exe"/>
            <uap5:ExecutionAlias Alias="backpropagate.exe"/>
          </uap5:AppExecutionAlias>
        </uap5:Extension>
      </Extensions>
    </Application>
  </Applications>
</Package>
"""


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)


def msix_version(pyproject_version: str) -> str:
    """pyproject X.Y.Z -> MSIX X.Y.Z.0 (the 4th part is reserved for the Store)."""
    parts = pyproject_version.split(".")
    if (
        len(parts) != 3
        or not all(p.isdigit() for p in parts)
        or int(parts[0]) == 0
    ):
        raise ValueError(
            f"pyproject version {pyproject_version!r} is not a plain X.Y.Z with "
            "a non-zero major; refusing to map it to an MSIX version"
        )
    return f"{pyproject_version}.0"


def filter_requirements(export_text: str) -> str:
    """Drop the torch stanza from a `uv export` requirements file.

    The lock's torch (PyPI) is CPU-only on win_amd64; the pinned cu130 wheel
    is installed separately. Stanza = a top-level requirement line plus its
    indented hash/comment continuation lines. nvidia-* entries carry
    ``sys_platform == 'linux'`` markers, so they never install on Windows —
    no filtering needed for them.
    """
    kept: list[str] = []
    skipping = False
    for line in export_text.splitlines():
        is_toplevel = line and not line[0].isspace() and not line.startswith("#")
        if is_toplevel:
            skipping = line.startswith("torch==")
        if not skipping:
            kept.append(line)
    return "\n".join(kept) + "\n"


def windowsapps_prefix(msix_ver: str) -> str:
    """The install prefix MSIX deploys to (no trailing separator).

    A fixed Windows target path by construction - built as a plain string
    so the value is identical no matter which OS runs the builder or tests.
    """
    return (
        f"C:\\Program Files\\WindowsApps\\{IDENTITY_NAME}_{msix_ver}_x64__{PUBLISHER_ID_SUFFIX}"
    )


def check_max_path(stage_root: Path, msix_ver: str) -> tuple[int, int]:
    """Return (deepest_package_relative_length, total_under_prefix); raise over budget."""
    deepest_rel, deepest_len = "", 0
    for file in stage_root.rglob("*"):
        if file.is_file():
            rel = len(str(file.relative_to(stage_root)))
            if rel > deepest_len:
                deepest_rel, deepest_len = str(file.relative_to(stage_root)), rel
    total = len(windowsapps_prefix(msix_ver)) + 1 + deepest_len
    if total > MAX_PATH_LIMIT:
        raise RuntimeError(
            f"MAX_PATH gate: {deepest_rel!r} is {deepest_len} chars inside the "
            f"package; under {windowsapps_prefix(msix_ver)!r} that is {total} "
            f"chars > {MAX_PATH_LIMIT} (LongPathsEnabled=0 machines)."
        )
    return deepest_len, total


def check_size(total_bytes: int, cap: int = SIZE_CAP_BYTES) -> None:
    if total_bytes > cap:
        raise RuntimeError(
            f"staged tree is {total_bytes / 1024**3:.1f} GiB > {cap / 1024**3:.0f} GiB cap"
        )


def render_manifest(msix_ver: str) -> str:
    return _MANIFEST_TEMPLATE.format(
        identity_name=IDENTITY_NAME,
        publisher=IDENTITY_PUBLISHER,
        msix_version=msix_ver,
        display_name=DISPLAY_NAME,
        publisher_display=PUBLISHER_DISPLAY_NAME,
    )


def torch_gate_script() -> str:
    """Python source run with the staging interpreter: hard CUDA proof."""
    return (
        "import json, torch\n"
        "cuda = torch.version.cuda or '0'\n"
        "v = tuple(int(p) for p in cuda.split('.')[:2])\n"
        "assert v >= (13, 0), f'torch CUDA variant too old: {cuda}'\n"
        "assert torch.cuda.is_available(), 'CUDA not available on the build GPU'\n"
        "x = torch.randn(64, device='cuda')\n"
        "_ = float((x * x).sum())\n"
        "print(json.dumps({'torch': torch.__version__, 'cuda': cuda, "
        "'device': torch.cuda.get_device_name(0), "
        "'capability': list(torch.cuda.get_device_capability(0))}))\n"
    )


# --------------------------------------------------------------------------
# Build steps


def _log(msg: str) -> None:
    print(f"[build-msix] {msg}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, sha256: str, *, label: str) -> Path:
    """Download with hash-pin; reuse a verified cache hit in dest."""
    if dest.is_file():
        if _sha256_file(dest) == sha256:
            _log(f"{label}: cached {dest.name} (sha256 verified)")
            return dest
        _log(f"{label}: cache hash mismatch -- refetching")
        dest.unlink()
    _log(f"{label}: fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as resp:  # noqa: S310 — pinned https URL + hash-checked below  # nosec B310
        with tmp.open("wb") as handle:
            shutil.copyfileobj(resp, handle, 16 * 1024 * 1024)
    actual = _sha256_file(tmp)
    if actual != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{label}: SHA-256 mismatch ({actual} != {sha256}) for {url}")
    os.replace(tmp, dest)
    _log(f"{label}: {dest.name} ({dest.stat().st_size / 1024**2:.0f} MiB, hash OK)")
    return dest


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    _log("run: " + " ".join(cmd)[:200])
    return subprocess.run(cmd, check=True, **kwargs)  # nosec B603 — fixed internal argv


def _project_version(repo: Path) -> str:
    import tomllib

    return tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]


def _find_sdk_tool(name: str) -> Path:
    kits = Path("C:/Program Files (x86)/Windows Kits/10/bin")
    if not kits.is_dir():
        raise RuntimeError(f"Windows SDK not found under {kits}")
    best: tuple[tuple[int, ...], Path] | None = None
    for candidate in kits.glob("*/x64/" + name):
        try:
            ver = tuple(int(p) for p in candidate.parent.parent.name.split("."))
        except ValueError:
            continue
        if best is None or ver > best[0]:
            best = (ver, candidate)
    if best is None:
        raise RuntimeError(f"{name} not found under {kits}/*/x64")
    return best[1]


def _find_csc() -> Path:
    for root in (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET/Framework64",
    ):
        candidates = sorted(root.glob("v*/csc.exe"), reverse=True)
        if candidates:
            return candidates[0]
    raise RuntimeError("no framework64 csc.exe found")


def stage_python(downloads: Path, stage: Path) -> Path:
    """Embedded CPython with a working ._pth; returns python.exe path."""
    zip_path = _fetch(PYTHON_EMBED_URL, downloads / f"python-{PYTHON_VERSION}-embed-amd64.zip", PYTHON_EMBED_SHA256, label="cpython-embed")
    py_home = stage / "App" / "python"
    if py_home.exists():
        shutil.rmtree(py_home)
    py_home.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(py_home)
    (py_home / "python312._pth").write_text(_PTH, encoding="utf-8")
    exe = py_home / "python.exe"
    result = _run([str(exe), "-c", "import sys, site; print(sys.version)"], capture_output=True, text=True)
    _log(f"embedded python: {result.stdout.strip().splitlines()[0]}")
    return exe


def install_dependencies(repo: Path, downloads: Path, stage: Path, python_exe: Path) -> dict:
    """Lock-closure minus PyPI torch, plus the pinned cu130 wheel, plus our own wheel."""
    export_path = downloads / "requirements-export.txt"
    _run(
        ["uv", "export", "--frozen", "--no-emit-project", "--extra", "ui",
         "--format", "requirements-txt", "-o", str(export_path)],
        cwd=str(repo),
    )
    filtered = filter_requirements(export_path.read_text(encoding="utf-8"))
    filtered_path = downloads / "requirements-filtered.txt"
    filtered_path.write_text(filtered, encoding="utf-8")
    kept_torch = [ln for ln in filtered.splitlines() if ln.startswith("torch==")]
    if kept_torch:
        raise RuntimeError("torch stanza survived the export filter — refusing")
    _run(
        ["uv", "pip", "install", "--python", str(python_exe), "--no-deps",
         "--require-hashes", "-r", str(filtered_path)],
        cwd=str(repo),
    )
    wheel = _fetch(TORCH_WHEEL_URL, downloads / "torch-2.12.1+cu130-cp312-cp312-win_amd64.whl", TORCH_WHEEL_SHA256, label="torch-cu130")
    _run(
        ["uv", "pip", "install", "--python", str(python_exe), "--no-deps", str(wheel)],
    )
    wheel_out = downloads / "dist-out"
    if wheel_out.exists():
        shutil.rmtree(wheel_out)
    _run(["uv", "build", "--wheel", "--out-dir", str(wheel_out)], cwd=str(repo))
    wheels = list(wheel_out.glob("backpropagate-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one built wheel, got {wheels}")
    _run(
        ["uv", "pip", "install", "--python", str(python_exe), "--no-deps", str(wheels[0])],
    )
    _run(["uv", "pip", "check", "--python", str(python_exe)])
    site_packages = stage / "App" / "python" / "Lib" / "site-packages"
    (site_packages / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return {"built_wheel": wheels[0].name}


def gate_torch(python_exe: Path) -> dict:
    result = _run(
        [str(python_exe), "-c", torch_gate_script()], capture_output=True, text=True
    )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    _log(f"torch gate: {info}")
    return info


def stage_payload(repo: Path, stage: Path, python_exe: Path, work: Path, reuse: Path | None) -> dict:
    if reuse is not None:
        _log(f"payload: reusing {reuse}")
        shutil.copytree(reuse, work / "ui_frontend_payload")
    else:
        result = _run(
            [str(python_exe), str(repo / "scripts" / "build_ui_frontend.py"), "--out", str(work)],
            cwd=str(repo),
        )
        del result
    payload = work / "ui_frontend_payload"
    meta = json.loads((payload / "payload.json").read_text(encoding="utf-8"))
    target = stage / "App" / "ui_frontend_payload"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(payload, target)
    return {"payload": meta}


def stage_llamacpp(downloads: Path, stage: Path) -> dict:
    archive = downloads / f"llama.cpp-{LLAMACPP_TAG}.zip"
    if not archive.is_file():
        # Tags move; verify the pin still resolves to the recorded commit.
        with urllib.request.urlopen(  # noqa: S310 — fixed https URL  # nosec B310
            f"https://api.github.com/repos/ggml-org/llama.cpp/git/refs/tags/{LLAMACPP_TAG}",
            timeout=60,
        ) as resp:
            sha = json.loads(resp.read().decode())["object"]["sha"]
        if sha != LLAMACPP_COMMIT:
            raise RuntimeError(
                f"llama.cpp tag {LLAMACPP_TAG} moved: {sha} != pinned {LLAMACPP_COMMIT}"
            )
        _log(f"llama.cpp: fetching archive {LLAMACPP_TAG} (commit verified)")
        with urllib.request.urlopen(LLAMACPP_ARCHIVE_URL, timeout=300) as resp:  # noqa: S310  # nosec B310 — tag pinned + per-file hashes verified below
            data = resp.read()
        archive.write_bytes(data)
    root_prefix = f"llama.cpp-{LLAMACPP_TAG}/"
    vendor = stage / "App" / "vendor" / "llama.cpp"
    if vendor.exists():
        shutil.rmtree(vendor)
    (vendor / "gguf-py").mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        converter = zf.read(root_prefix + "convert_hf_to_gguf.py")
        gguf_init = zf.read(root_prefix + "gguf-py/gguf/__init__.py")
        license_bytes = zf.read(root_prefix + "LICENSE")
        (vendor / "convert_hf_to_gguf.py").write_bytes(converter)
        (vendor / "LICENSE").write_bytes(license_bytes)
        for name in zf.namelist():
            if name.startswith(root_prefix + "gguf-py/gguf/") and not name.endswith("/"):
                rel = name[len(root_prefix):]
                dest = vendor / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))
    if hashlib.sha256(converter).hexdigest() != LLAMACPP_CONVERTER_SHA256:
        raise RuntimeError("llama.cpp convert_hf_to_gguf.py hash drifted vs the pin")
    if hashlib.sha256(gguf_init).hexdigest() != LLAMACPP_GGUF_INIT_SHA256:
        raise RuntimeError("llama.cpp gguf-py/gguf/__init__.py hash drifted vs the pin")
    _log(f"llama.cpp: {LLAMACPP_TAG} staged (per-file hashes verified)")
    return {"llama_cpp": {"tag": LLAMACPP_TAG, "commit": LLAMACPP_COMMIT}}


def stage_launcher(stage: Path, downloads: Path) -> None:
    csc = _find_csc()
    src = downloads / "backprop_launcher.cs"
    src.write_text(_LAUNCHER_CS, encoding="utf-8")
    out = stage / "App" / "backprop-launcher.exe"
    _run([str(csc), "/nologo", "/target:exe", f"/out:{out}", str(src)])


def stage_assets(repo: Path, stage: Path) -> None:
    from PIL import Image

    src = Image.open(repo / "assets" / "logo.png").convert("RGBA")
    bg = src.getpixel((0, 0))
    if bg[3] == 0:
        bg = (0x0B, 0x0E, 0x14, 255)  # dark fallback when the logo is transparent-edged
    canvas_w = max(src.size)
    canvas = Image.new("RGBA", (canvas_w, canvas_w), bg)
    canvas.paste(src, ((canvas_w - src.width) // 2, (canvas_w - src.height) // 2), src)
    assets = stage / "Assets"
    assets.mkdir(exist_ok=True)
    targets = [
        ("StoreLogo.png", (50, 50)),
        ("Square44x44Logo.png", (44, 44)),
        ("Square71x71Logo.png", (71, 71)),
        ("Square150x150Logo.png", (150, 150)),
        ("Square310x310Logo.png", (310, 310)),
        ("Wide310x150Logo.png", (310, 150)),
        ("Square44x44Logo.targetsize-256.png", (256, 256)),
        ("Square44x44Logo.targetsize-48.png", (48, 48)),
        ("Square44x44Logo.targetsize-32.png", (32, 32)),
        ("Square44x44Logo.targetsize-24.png", (24, 24)),
        ("Square44x44Logo.targetsize-16.png", (16, 16)),
    ]
    for name, size in targets:
        resized = canvas.resize(size, Image.LANCZOS)
        resized.convert("RGB").save(assets / name) if name != "StoreLogo.png" else resized.save(assets / name)
    _log(f"assets: {len(targets)} logos generated (pad color {bg})")


def stage_notices(repo: Path, stage: Path) -> None:
    parts: list[str] = []
    for title, source in NOTICES:
        if source is None and title.startswith("backpropagate"):
            text = (repo / "LICENSE").read_text(encoding="utf-8")
        elif source is None:  # llama.cpp — from the staged vendor copy
            text = (stage / "App" / "vendor" / "llama.cpp" / "LICENSE").read_text(encoding="utf-8")
        else:
            with urllib.request.urlopen(source, timeout=60) as resp:  # noqa: S310 — pinned tag URL  # nosec B310
                text = resp.read().decode("utf-8")
        parts.append(f"{'=' * 78}\n{title}\n{'=' * 78}\n\n{text.strip()}\n")
    (stage / "THIRD_PARTY_NOTICES.txt").write_text(
        "backpropagate Microsoft Store package — third-party notices\n\n"
        + "\n".join(parts),
        encoding="utf-8",
    )
    _log(f"notices: {len(NOTICES)} license texts")


def write_manifest_and_info(stage: Path, msix_ver: str, meta: dict, total_bytes: int, deepest: int, deepest_total: int) -> dict:
    (stage / "AppxManifest.xml").write_text(render_manifest(msix_ver), encoding="utf-8")
    info = {
        "schema": 1,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": {
            "name": IDENTITY_NAME,
            "publisher": IDENTITY_PUBLISHER,
            "publisher_display": PUBLISHER_DISPLAY_NAME,
            "package_family_suffix": PUBLISHER_ID_SUFFIX,
        },
        "version": meta["version"],
        "msix_version": msix_ver,
        "python": {
            "version": PYTHON_VERSION,
            "url": PYTHON_EMBED_URL,
            "sha256": PYTHON_EMBED_SHA256,
        },
        "torch": {
            "version": f"{TORCH_VERSION}+{TORCH_CU_VARIANT}",
            "url": TORCH_WHEEL_URL,
            "sha256": TORCH_WHEEL_SHA256,
            "min_nvidia_driver": MIN_NVIDIA_DRIVER,
            "gate": meta.get("torch_gate"),
        },
        "llama_cpp": {"tag": LLAMACPP_TAG, "commit": LLAMACPP_COMMIT},
        "ui_payload": meta.get("payload"),
        "built_wheel": meta.get("built_wheel"),
        "size_bytes": total_bytes,
        "maxpath": {"deepest_relative": deepest, "total_under_install_prefix": deepest_total},
    }
    (stage / "App" / "build-info.json").write_text(
        json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return info


def pack(stage: Path, out_dir: Path, unsigned_name: str) -> Path:
    makeappx = _find_sdk_tool("makeappx.exe")
    dest = out_dir / unsigned_name
    if dest.exists():
        dest.unlink()
    _run([str(makeappx), "pack", "/d", str(stage), "/p", str(dest), "/o"])
    _log(f"packed: {dest} ({dest.stat().st_size / 1024**3:.2f} GiB)")
    return dest


def sideload_sign(msix: Path) -> None:
    """Self-sign for LOCAL verification only. The cert subject matches the
    Partner Center publisher so the signature validates against the manifest.
    The signing key is destroyed immediately after signing (the .pfx is
    deleted and the cert is removed from Cert:\\CurrentUser\\My with
    -DeleteKey); only the public .cer remains, for the trust-store import
    printed below. The trust/import commands are for a human admin to run -
    this script never touches certificate stores beyond its own throwaway cert.
    """
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        raise RuntimeError("no pwsh/powershell found for self-signing")
    cert_dir = msix.parent / "sideload-cert"
    cert_dir.mkdir(exist_ok=True)
    pfx = cert_dir / "backpropagate-sideload.pfx"
    cer = cert_dir / "backpropagate-sideload.cer"
    password = "sideload-test"  # nosec B105 - throwaway password for the local self-signed test cert, not a shipped secret
    create = (
        "$p = ConvertTo-SecureString -String '" + password + "' -Force -AsPlainText; "
        f"$c = New-SelfSignedCertificate -Type Custom -Subject '{IDENTITY_PUBLISHER}' "
        "-KeyUsage DigitalSignature -FriendlyName 'backpropagate sideload test' "
        "-CertStoreLocation Cert:\\CurrentUser\\My "
        "-TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3'); "
        f"Export-PfxCertificate -Cert $c -FilePath '{pfx}' -Password $p | Out-Null; "
        f"Export-Certificate -Cert $c -FilePath '{cer}' | Out-Null; "
        "Write-Output $c.Thumbprint"
    )
    created = subprocess.run(  # nosec B603 - fixed internal argv
        [pwsh, "-NoProfile", "-Command", create], check=True, capture_output=True, text=True
    )
    thumbprints = re.findall(r"\b[0-9A-Fa-f]{40}\b", created.stdout)
    if not thumbprints:
        raise RuntimeError(
            "could not read the self-signed cert thumbprint from pwsh output: "
            f"{created.stdout!r}"
        )
    thumbprint = thumbprints[-1].upper()
    signtool = _find_sdk_tool("signtool.exe")
    _run([str(signtool), "sign", "/fd", "sha256", "/a", "/f", str(pfx), "/p", password, str(msix)])
    # Destroy the signing key: a usable key for the Store publisher CN must not
    # remain on disk. -DeleteKey wipes the key material with the cert.
    pfx.unlink(missing_ok=True)
    subprocess.run(  # nosec B603 - fixed internal argv; removes only the cert created above
        [pwsh, "-NoProfile", "-Command",
         f"Remove-Item 'Cert:\\CurrentUser\\My\\{thumbprint}' -DeleteKey -Force"],
        check=True, capture_output=True, text=True,
    )
    print(
        f"\nSIGNED FOR SIDELOAD TESTING ONLY (throwaway cert {thumbprint}; its private\n"
        "key was deleted right after signing and is unrecoverable).\n"
        "Have an admin run:\n"
        f"  certutil -addstore TrustedPeople \"{cer}\"\n"
        f"  Add-AppxPackage \"{msix}\"\n"
        "After testing, remove every trace:\n"
        "  Remove-AppxPackage -Package (Get-AppxPackage *backpropagate*).PackageFullName\n"
        f"  certutil -delstore TrustedPeople {thumbprint}\n"
        f"  Remove-Item \"{cer}\"\n"
        "The Store build is the UNSIGNED .msix -- Partner Center re-signs it.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, required=True, help="output dir for the msix + caches")
    parser.add_argument("--reuse-payload", type=Path, default=None,
                        help="reuse an existing ui_frontend_payload dir instead of rebuilding")
    parser.add_argument("--sideload-test", action="store_true",
                        help="self-sign the msix for LOCAL sideload verification (never for the Store upload)")
    parser.add_argument("--pack-only", type=Path, default=None,
                        help="skip staging; pack an existing stage dir (gates re-run)")
    args = parser.parse_args(argv)

    if os.name != "nt":
        print("build_msix.py must run on Windows", file=sys.stderr)
        return 2
    t0 = time.monotonic()
    repo = args.repo.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads = out_dir / "_downloads"
    downloads.mkdir(exist_ok=True)

    version = _project_version(repo)
    msix_ver = msix_version(version)
    _log(f"version {version} -> msix {msix_ver}")

    stage = out_dir / "stage"
    meta: dict = {"version": version}

    if args.pack_only is not None:
        stage = args.pack_only.resolve()
        if not stage.is_dir():
            print(f"--pack-only stage dir missing: {stage}", file=sys.stderr)
            return 2
        _log(f"pack-only: reusing stage {stage}")
    else:
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        python_exe = stage_python(downloads, stage)
        meta.update(install_dependencies(repo, downloads, stage, python_exe))
        meta["torch_gate"] = gate_torch(python_exe)
        meta.update(stage_payload(repo, stage, python_exe, out_dir / "_payload", args.reuse_payload))
        meta.update(stage_llamacpp(downloads, stage))
        stage_launcher(stage, downloads)
        stage_assets(repo, stage)
        stage_notices(repo, stage)

    total_bytes = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    check_size(total_bytes)
    deepest, deepest_total = check_max_path(stage, msix_ver)
    info = write_manifest_and_info(stage, msix_ver, meta, total_bytes, deepest, deepest_total)
    (out_dir / "build-info.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    _log("size report:")
    for child in sorted(stage.iterdir()):
        if child.is_dir():
            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            print(f"  {child.name + '/':<28} {size / 1024**2:>9.0f} MiB", flush=True)
        else:
            size = child.stat().st_size
            if size > 1 << 20:
                print(f"  {child.name:<28} {size / 1024**2:>9.0f} MiB", flush=True)
    print(f"  {'TOTAL':<28} {total_bytes / 1024**3:>9.2f} GiB (cap {SIZE_CAP_BYTES / 1024**3:.0f} GiB)", flush=True)
    print(f"  MAX_PATH deepest: {deepest_total} chars incl. install prefix (limit {MAX_PATH_LIMIT})", flush=True)

    msix = pack(stage, out_dir, f"backpropagate_{msix_ver}_x64.msix")
    if args.sideload_test:
        sideload_sign(msix)
    _log(f"done in {(time.monotonic() - t0) / 60:.1f} min -> {msix}")
    _log("next: WACK (appcert.exe validate), sideload verification, then Partner Center upload of the UNSIGNED build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
