"""Real-GPU smoke of the golden path: QLoRA 4-bit SFT -> export -> GGUF -> Ollama.

This smoke tests the product's headline promise ("train, then one-click GGUF
export to Ollama") on real hardware. It uses no mocks. The only other
end-to-end export test (``tests/test_e2e_chain.py``) writes a stub GGUF, so it
cannot see bugs in the real export chain. This one reproduces the exact shape
of #132 and #133:

1. **Train.** ``Trainer`` with its defaults (QLoRA: 4-bit nf4 base + LoRA; Unsloth
   if installed, else transformers + PEFT) runs 2 SFT steps on
   ``examples/quickstart.jsonl``. The HF trainer writes ``output/checkpoint-2``,
   the directory a user hands to ``backprop export``.
2. **Export from the saved checkpoint, through the CLI.** It runs
   ``python -m backpropagate export <checkpoint> --format ...`` in a
   subprocess, the same entry point as the ``backprop`` console script. It does
   not export from the in-memory trainer, because #132 only happened when the
   CLI reloaded a checkpoint from disk.
3. **Check that the GGUF is real.** The file must start with the ``GGUF``
   magic, have a sane version, and hold tensors plus a KV table that parses.
   ``general.architecture`` must be present.
4. **Ollama.** If a daemon is reachable, the GGUF is registered under a
   throwaway name, a few tokens are generated, and the model is removed. If no
   daemon is reachable, the test skips and says why.

Tests
-----
* ``test_merged_export_from_checkpoint`` is the #132 regression. It runs
  ``--format merged`` from the checkpoint and needs no GGUF tooling, so it runs
  wherever CUDA and the model are available.
* ``test_gguf_export_llama_cpp_fallback`` covers the #133 path. It runs
  ``--format gguf --quantization q8_0`` through llama.cpp's
  ``convert_hf_to_gguf.py``. It skips unless ``BACKPROPAGATE_LLAMA_CPP_PATH``
  resolves to the script, and asserts ``general.name`` is not derived from the
  temp directory name.
* ``test_gguf_export_unsloth`` covers the default ``q4_k_m`` path through
  Unsloth's ``save_pretrained_gguf``. It skips unless Unsloth is installed
  and Unsloth's own llama.cpp build (``~/.unsloth/llama.cpp`` or
  ``UNSLOTH_LLAMA_CPP_PATH``) is already in place. The export subprocess sets
  ``UNSLOTH_AUTO_INSTALL=0``, because on Windows Unsloth would otherwise
  ``winget install`` CMake, OpenSSL and VS Build Tools, accepting their
  licence agreements. A smoke must never change the host system.
* ``test_ollama_roundtrip`` registers whichever GGUF the run produced.

Gating
------
The tests are marked ``slow`` and ``integration``, so the fast lane deselects
them. They skip, and name the fix, when a training dependency is missing, when
CUDA is absent (bitsandbytes 4-bit needs CUDA), or when the model is not
cached and the network is unreachable. Run the whole battery with
``bash scripts/gpu_smoke.sh``, or run just this smoke with
``pytest tests/test_golden_path_smoke.py -m "slow or integration" -p no:randomly --timeout=0``.

HF cache: if the model sits in ``~/.cache/huggingface/hub`` but ``HF_HOME`` points
elsewhere, set ``HF_HUB_CACHE`` to that directory. ``scripts/gpu_smoke.sh``
does this automatically.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest

_SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUICKSTART = _REPO_ROOT / "examples" / "quickstart.jsonl"
_EXPORT_TIMEOUT_S = 1800


def _model_is_reachable(model_id: str) -> bool:
    """True if ``model_id`` can be loaded — network reachable OR already cached."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST

        cached = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached, str) and cached and cached is not _CACHED_NO_EXIST:
            return True
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _unsloth_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("unsloth") is not None


def _llama_cpp_convert_script() -> Path | None:
    """The converter ``BACKPROPAGATE_LLAMA_CPP_PATH`` points at, if it exists."""
    raw = os.environ.get("BACKPROPAGATE_LLAMA_CPP_PATH")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_dir():
        path = path / "convert_hf_to_gguf.py"
    return path if path.is_file() else None


def _unsloth_llama_cpp_ready() -> tuple[bool, str]:
    """Whether Unsloth's GGUF path can run WITHOUT installing anything."""
    try:
        from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR, check_llama_cpp
    except Exception as e:  # pragma: no cover - environment-dependent
        return False, f"unsloth_zoo.llama_cpp not importable ({e})"
    try:
        check_llama_cpp(LLAMA_CPP_DEFAULT_DIR)
    except Exception as e:
        return False, (
            f"Unsloth's llama.cpp build is not in place at {LLAMA_CPP_DEFAULT_DIR} "
            f"({str(e).splitlines()[0]}). Build llama.cpp there (or set "
            "UNSLOTH_LLAMA_CPP_PATH) and re-run; the smoke will not let Unsloth "
            "auto-install system packages."
        )
    return True, ""


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def _ollama_reachable() -> bool:
    if shutil.which("ollama") is None:
        return False
    try:
        with urllib.request.urlopen(f"{_ollama_host()}/api/version", timeout=3) as r:  # noqa: S310 - local daemon
            return r.status == 200
    except Exception:
        return False


_MISSING_DEPS: list[str] = []
for _dep in ("torch", "trl", "transformers", "peft", "datasets", "bitsandbytes"):
    try:
        __import__(_dep)
    except Exception:  # pragma: no cover - environment-dependent
        _MISSING_DEPS.append(_dep)

_SKIP_REASON: str | None = None
if _MISSING_DEPS:
    _SKIP_REASON = (
        f"golden-path smoke requires {', '.join(_MISSING_DEPS)} "
        "(pip install 'backpropagate[standard]' or plain `pip install backpropagate`)"
    )
elif not _cuda_available():
    _SKIP_REASON = (
        "golden-path smoke trains QLoRA on a bitsandbytes 4-bit base, which needs "
        "a CUDA GPU. Run it on a CUDA box (e.g. the RTX 5090 rig)."
    )
elif not _model_is_reachable(_SMOKE_MODEL):
    _SKIP_REASON = (
        f"{_SMOKE_MODEL} is not reachable (no network AND not in the HF cache). "
        f"Run `huggingface-cli download {_SMOKE_MODEL}`, or point HF_HUB_CACHE at "
        "the cache that holds it, then re-run."
    )

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or ""),
]


# ---------------------------------------------------------------------------
# GGUF header reader (no dependency on the gguf package)
# ---------------------------------------------------------------------------

_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def _read_gguf_header(path: Path) -> dict[str, Any]:
    """Parse a GGUF header and KV table, and return the KVs plus counts.

    Arrays are skipped past but not returned. Only scalars and strings are
    kept. Raises ``AssertionError`` on anything that is not a well-formed
    GGUF v2/v3 header.
    """
    with open(path, "rb") as fh:

        def take(n: int) -> bytes:
            b = fh.read(n)
            assert len(b) == n, f"{path}: truncated GGUF header"
            return b

        def u32() -> int:
            return int(struct.unpack("<I", take(4))[0])

        def u64() -> int:
            return int(struct.unpack("<Q", take(8))[0])

        def string() -> str:
            return take(u64()).decode("utf-8", errors="replace")

        def value(vtype: int) -> Any:
            if vtype == _GGUF_STRING:
                return string()
            if vtype == _GGUF_ARRAY:
                etype, count = u32(), u64()
                for _ in range(count):
                    value(etype)
                return None
            fmt = _GGUF_SCALAR.get(vtype)
            assert fmt is not None, f"{path}: unknown GGUF value type {vtype}"
            return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

        magic = take(4)
        assert magic == b"GGUF", f"{path}: bad magic {magic!r} (not a GGUF file)"
        version = u32()
        assert version in (2, 3), f"{path}: unexpected GGUF version {version}"
        tensor_count, kv_count = u64(), u64()
        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = string()
            kv[key] = value(u32())
    return {"version": version, "tensor_count": tensor_count, "kv": kv}


def _assert_real_gguf(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"GGUF not written: {path}"
    size = path.stat().st_size
    # A stub (b"GGUF...MOCK") is a few bytes; SmolLM2-135M at q8_0 is ~140 MB.
    assert size > 10 * 1024 * 1024, f"GGUF implausibly small ({size} bytes): {path}"
    header = _read_gguf_header(path)
    assert header["tensor_count"] > 0, f"GGUF has no tensors: {path}"
    assert header["kv"].get("general.architecture"), (
        f"GGUF lacks general.architecture: {sorted(header['kv'])[:20]}"
    )
    return header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``python -m backpropagate <args>`` — the ``backprop`` entry point."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # Never let Unsloth mutate the host (winget / apt installs) during a smoke.
    env["UNSLOTH_AUTO_INSTALL"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [sys.executable, "-m", "backpropagate", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_EXPORT_TIMEOUT_S,
    )


def _assert_cli_ok(proc: subprocess.CompletedProcess[str], what: str) -> str:
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, (
        f"{what} exited {proc.returncode}. Output tail:\n{out[-4000:]}"
    )
    # #132's fingerprints: the loader must not stack a second PEFT wrapper.
    assert "active_adapters" not in out, out[-4000:]
    assert "modify a model with PEFT for a second time" not in out, out[-4000:]
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("golden_path")


@pytest.fixture(scope="module")
def checkpoint(workdir: Path) -> Path:
    """QLoRA 4-bit SFT, 2 steps, with the Trainer's defaults. Returns checkpoint-N."""
    import gc

    import torch

    from backpropagate.trainer import Trainer, TrainingRun

    assert _QUICKSTART.is_file(), f"missing {_QUICKSTART}"
    output_dir = workdir / "output"
    trainer = Trainer(
        model=_SMOKE_MODEL,
        max_seq_length=256,
        lora_r=8,
        batch_size=1,
        gradient_accumulation=1,
        output_dir=str(output_dir),
        report_to="none",
    )
    run = trainer.train(str(_QUICKSTART), steps=2)
    assert isinstance(run, TrainingRun)
    assert run.final_loss is not None and math.isfinite(run.final_loss), (
        f"final_loss must be finite; got {run.final_loss!r}"
    )
    # The golden path is QLoRA: the trained base must be bitsandbytes 4-bit.
    assert any(type(m).__name__ == "Linear4bit" for m in trainer.model.modules()), (
        "expected a bitsandbytes 4-bit base (QLoRA default); found no Linear4bit"
    )

    ckpts = sorted(output_dir.glob("checkpoint-*"))
    assert ckpts, f"trainer wrote no checkpoint-N under {output_dir}"
    ckpt = ckpts[-1]
    assert (ckpt / "adapter_config.json").is_file(), f"{ckpt} is not a PEFT adapter dir"

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt


@pytest.fixture(scope="module")
def fallback_gguf(checkpoint: Path, workdir: Path) -> Path:
    script = _llama_cpp_convert_script()
    if script is None:
        pytest.skip(
            "llama.cpp fallback not configured: set BACKPROPAGATE_LLAMA_CPP_PATH to a "
            "llama.cpp source clone (or its convert_hf_to_gguf.py), with llama.cpp's "
            "converter requirements (sentencepiece, protobuf) installed in this venv."
        )
    out_dir = workdir / "gguf_fallback"
    proc = _run_cli(
        ["export", str(checkpoint), "--format", "gguf", "--quantization", "q8_0",
         "--output", str(out_dir)],
        cwd=workdir,
    )
    _assert_cli_ok(proc, "backprop export --format gguf --quantization q8_0")
    ggufs = sorted(out_dir.glob("*.gguf"))
    assert ggufs, f"no .gguf in {out_dir}"
    return ggufs[0]


@pytest.fixture(scope="module")
def unsloth_gguf(checkpoint: Path, workdir: Path) -> Path:
    if not _unsloth_installed():
        pytest.skip("Unsloth not installed; its save_pretrained_gguf path is not exercised.")
    ready, why = _unsloth_llama_cpp_ready()
    if not ready:
        pytest.skip(why)
    out_dir = workdir / "gguf_unsloth"
    proc = _run_cli(
        ["export", str(checkpoint), "--format", "gguf", "--quantization", "q4_k_m",
         "--output", str(out_dir)],
        cwd=workdir,
        # Keep the llama.cpp fallback out of it: this fixture measures Unsloth.
        extra_env={"BACKPROPAGATE_LLAMA_CPP_PATH": str(workdir / "no-llama-cpp")},
    )
    _assert_cli_ok(proc, "backprop export --format gguf --quantization q4_k_m (unsloth)")
    ggufs = sorted(out_dir.glob("*.gguf"))
    assert ggufs, f"no .gguf in {out_dir}"
    return ggufs[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_merged_export_from_checkpoint(checkpoint: Path, workdir: Path) -> None:
    """#132 regression: export a saved bnb-4bit QLoRA checkpoint through the CLI.

    Before the fix this died with ``UnboundLocalError: ... 'active_adapters'``
    after PEFT warned it was modifying the model a second time. The merged
    checkpoint must also be a plain 16-bit model. A bitsandbytes-quantized
    merge cannot be converted to GGUF.
    """
    out_dir = workdir / "merged"
    proc = _run_cli(
        ["export", str(checkpoint), "--format", "merged", "--output", str(out_dir)],
        cwd=workdir,
    )
    _assert_cli_ok(proc, "backprop export --format merged")

    assert (out_dir / "model.safetensors").is_file() or list(
        out_dir.glob("model-*.safetensors")
    ), f"no merged weights in {out_dir}: {sorted(p.name for p in out_dir.iterdir())}"
    assert not (out_dir / "adapter_config.json").exists(), (
        "merged export wrote an adapter, not a merged model"
    )
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config, (
        "merged checkpoint is bitsandbytes-quantized; the merge must happen on a "
        f"16-bit base. quantization_config={config.get('quantization_config')!r}"
    )


def test_gguf_export_llama_cpp_fallback(fallback_gguf: Path) -> None:
    """#133: the llama.cpp fallback writes a real GGUF with a sane general.name."""
    header = _assert_real_gguf(fallback_gguf)
    name = header["kv"].get("general.name")
    # Without --model-name the converter titles the temp dir ("Merged_Temp").
    assert name and "merged" not in str(name).lower(), (
        f"general.name derived from the temp dir: {name!r}"
    )


def test_gguf_export_unsloth(unsloth_gguf: Path) -> None:
    """The default q4_k_m path through Unsloth's save_pretrained_gguf."""
    _assert_real_gguf(unsloth_gguf)


def test_ollama_roundtrip(request: pytest.FixtureRequest) -> None:
    """Register the exported GGUF with Ollama, generate a few tokens, remove it."""
    if not _ollama_reachable():
        pytest.skip(
            f"Ollama daemon not reachable at {_ollama_host()} (or `ollama` not on "
            "PATH); start it with `ollama serve` to run the registration stage."
        )
    gguf: Path | None = None
    skipped: list[str] = []
    for fixture_name in ("fallback_gguf", "unsloth_gguf"):
        try:
            gguf = request.getfixturevalue(fixture_name)
            break
        except pytest.skip.Exception as e:
            skipped.append(f"{fixture_name}: {e}")
    if gguf is None:
        pytest.skip("no GGUF backend available to produce a model for Ollama; " + " | ".join(skipped))

    from backpropagate.export import list_ollama_models, register_with_ollama, remove_ollama_model

    name = f"bp-golden-smoke-{uuid.uuid4().hex[:8]}"
    try:
        assert register_with_ollama(gguf, name) is True
        listed = list_ollama_models()
        assert any(m == name or m.startswith(f"{name}:") for m in listed), (
            f"{name} not in `ollama list`: {listed}"
        )
        body = json.dumps({
            "model": name,
            "prompt": "What is LoRA?",
            "stream": False,
            "options": {"num_predict": 8, "temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{_ollama_host()}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - local daemon
            reply = json.loads(r.read().decode("utf-8"))
        assert reply.get("done") is True, reply
        assert reply.get("eval_count", 0) > 0, f"Ollama generated no tokens: {reply}"
    finally:
        remove_ollama_model(name)
