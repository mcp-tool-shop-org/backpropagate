# VRAM calibration GATE: real GPU, real training probes, real browser.
"""``backprop estimate-vram --calibrate`` and "Measure on this GPU" for real.

Heavy and opt-in (``BACKPROPAGATE_UI_FLOW=1``, CUDA). Runs BY HAND on the dev
rig, never in CI, like the ui-v2 flow tests. Every job here is bounded: the
largest is Llama 3.2 1B at batch 4 x 2,048 tokens (about 10 GiB), and the
calibration itself refuses any probe not predicted to fit in free VRAM.

1. CLI: calibrate Llama 3.2 1B (QLoRA), then train one HELD-OUT config the
   probes did not run (batch 3 x 2,048 tokens) and compare its real peak with
   what the stored measurement predicts.
2. UI: press "Measure on this GPU" for SmolLM2-135M in a real browser; the job
   runs through the JobManager and the estimate then reads "measured".

Run (rig):

    BACKPROPAGATE_UI_FLOW=1 PYTHONPATH=. python -m pytest tests/test_vram_calibration_gpu.py -v -s

Last run on the RTX 5090 (2026-10-02): both passed in 3 min 58 s; the held-out
batch 3 x 2,048 run peaked at 7.47 GiB against 7.44 GiB predicted (-0.4%).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_CUDA = False
try:
    import torch

    _CUDA = bool(torch.cuda.is_available())
except Exception:  # noqa: BLE001
    _CUDA = False

PLAYWRIGHT_OK = True
try:
    from playwright.sync_api import sync_playwright
except Exception:  # noqa: BLE001
    PLAYWRIGHT_OK = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(3600),
    pytest.mark.skipif(not _CUDA, reason="no CUDA GPU"),
    pytest.mark.skipif(
        os.environ.get("BACKPROPAGATE_UI_FLOW") != "1",
        reason="set BACKPROPAGATE_UI_FLOW=1 to opt in (real GPU training)",
    ),
]

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"

_HELD_OUT = r'''
import json, os, sys, tempfile
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "disabled")
import torch
from backpropagate.trainer import Trainer
from backpropagate.vram_calibration import _probe_dataset
model, batch, seq = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
free, total = torch.cuda.mem_get_info()
torch.cuda.set_per_process_memory_fraction(min(0.98, (free / total) - 0.03))
tmp = Path(tempfile.mkdtemp(prefix="bp-heldout-"))
t = Trainer(model=model, batch_size=batch, max_seq_length=seq, output_dir=str(tmp / "o"),
            report_to="none", oom_recovery=False, lora_r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"])
t.load_model()
torch.cuda.reset_peak_memory_stats()
t.train(dataset=str(_probe_dataset(tmp, seq, 4 * batch)), steps=2)
print("HELDOUT " + json.dumps({"peak_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
'''


def _env(store: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
    env["BACKPROPAGATE_VRAM_CALIBRATION"] = str(store)
    env["PYTHONIOENCODING"] = "utf-8"
    env["WANDB_MODE"] = "disabled"
    return env


def _last_json(out: str, key: str) -> dict:
    decoder, i, found = json.JSONDecoder(), 0, None
    while (i := out.find("{", i)) != -1:
        try:
            obj, _end = decoder.raw_decode(out, i)
            if isinstance(obj, dict) and key in obj:
                found = obj
        except json.JSONDecodeError:
            pass
        i += 1
    assert found is not None, out[-3000:]
    return found


def test_cli_calibration_predicts_a_held_out_run(tmp_path):
    store = tmp_path / "cal.json"
    env = _env(store)
    proc = subprocess.run(
        [sys.executable, "-m", "backpropagate", "estimate-vram", LLAMA_1B, "--calibrate", "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO_ROOT), env=env, timeout=1800,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    cal = _last_json(proc.stdout, "calibration")["calibration"]
    assert store.exists()
    assert cal["quad_bytes"] is not None and cal["max_residual_pct"] < 10
    assert len([p for p in cal["probes"] if not p.get("oom")]) >= 2
    assert 0.5 < cal["load_gib"] < 2.0 and 0.9 < cal["floor_gib"] < 1.1

    # The estimate for a config the probes did not run, from the measurement.
    est = subprocess.run(
        [sys.executable, "-m", "backpropagate", "estimate-vram", LLAMA_1B, "--lora-r", "16",
         "--target-modules", "q_proj,v_proj", "--batch-size", "3", "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO_ROOT), env=env, timeout=600,
    )
    per = _last_json(est.stdout, "per_config_estimate")["per_config_estimate"]
    assert per["source"] == "measured"
    predicted = float(per["total_gb"]) - float(per["overhead_gb"])  # peak allocation

    held = subprocess.run(
        [sys.executable, "-c", _HELD_OUT, LLAMA_1B, "3", "2048"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(REPO_ROOT), env=env, timeout=1800,
    )
    line = [ln for ln in held.stdout.splitlines() if ln.startswith("HELDOUT ")]
    assert line, (held.stdout + held.stderr)[-3000:]
    measured = json.loads(line[-1][len("HELDOUT "):])["peak_gib"]
    print(f"\nheld-out batch 3 x 2048: measured {measured:.2f} GiB, predicted {predicted:.2f} GiB "
          f"({100 * (predicted - measured) / measured:+.1f}%)")
    assert predicted == pytest.approx(measured, rel=0.10)


@pytest.mark.skipif(not PLAYWRIGHT_OK, reason="playwright not installed")
def test_measure_on_this_gpu_button(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    from test_ui_e2e_real_reflex import _banner_token, _base_port, _UiLaunch

    store = tmp_path / "cal.json"
    out_dir = tmp_path / "ui-outputs"
    launch = _UiLaunch(
        tmp_path, _base_port(), [],
        extra_env={
            "BACKPROPAGATE_UI__OUTPUT_DIR": str(out_dir),
            "BACKPROPAGATE_VRAM_CALIBRATION": str(store),
        },
    )
    try:
        launch.wait_ready()
        token = _banner_token(launch)
        base = f"http://127.0.0.1:{launch.port}"
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(f"{base}/?token={token}")
            page.wait_for_load_state("networkidle")
            page.get_by_label("HuggingFace model id").fill(SMALL)
            page.get_by_text("VRAM · estimate").first.wait_for(timeout=90_000)
            page.get_by_role("button", name="Measure on this GPU").click()
            page.get_by_text("MEASURING").first.wait_for(timeout=60_000)
            shot = os.environ.get("BACKPROPAGATE_UI_SHOT_DIR", "").strip()
            if shot:
                Path(shot).mkdir(parents=True, exist_ok=True)
                time.sleep(2)
                page.screenshot(path=str(Path(shot) / "measuring.png"))
            page.get_by_text("VRAM · measured on this GPU").first.wait_for(timeout=900_000)
            page.get_by_role("button", name="Measure again").first.wait_for(timeout=30_000)
            if shot:
                time.sleep(2)
                page.screenshot(path=str(Path(shot) / "measured.png"))
            browser.close()
        data = json.loads(store.read_text(encoding="utf-8"))
        assert len(data) == 1
        entry = next(iter(data.values()))
        assert entry["model"] == SMALL and entry["load_gib"] > 0
        job = sorted((out_dir / "jobs").glob("run_*"))[-1]
        argv = json.loads((job / "job.json").read_text(encoding="utf-8"))["argv_tail"]
        assert argv[0] == "--calibrate" and argv[-2:] == ["--", SMALL]  # the model follows "--"
    finally:
        launch.stop()
