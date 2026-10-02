# ui-v2 P1 GATE: real-browser, real-GPU training flow through the web UI.
"""End-to-end: UI -> JobManager -> subprocess -> Stop-and-save -> reattach.

This is the P1 acceptance test for ui-v2 (docs/handoff-2026-10-01-ui-v2.md).
It is heavy: a real ``backprop ui`` server, the real Chrome browser, and a
real ~20-step training run of a ~135M model on the local GPU. It runs BY HAND
on the dev rig, never in CI — same doctrine as tests/test_*_smoke.py.

Gating:

* ``playwright`` + Chrome installed (scratch venv),
* CUDA available,
* set BACKPROPAGATE_UI_FLOW=1 to opt in.

Run (rig):

    PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/test_ui_v2_p1_flow.py -v -s
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PLAYWRIGHT_OK = True
try:
    from playwright.sync_api import sync_playwright
except Exception:  # noqa: BLE001
    PLAYWRIGHT_OK = False

_CUDA = False
try:
    import torch

    _CUDA = bool(torch.cuda.is_available())
except Exception:  # noqa: BLE001
    _CUDA = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not PLAYWRIGHT_OK, reason="playwright not installed"),
    pytest.mark.skipif(not _CUDA, reason="no CUDA GPU"),
    pytest.mark.skipif(
        os.environ.get("BACKPROPAGATE_UI_FLOW") != "1",
        reason="set BACKPROPAGATE_UI_FLOW=1 to opt in (real GPU training)",
    ),
]

sys.path.insert(0, str(Path(__file__).parent))
from test_ui_e2e_real_reflex import (  # noqa: E402
    _banner_token,
    _base_port,
    _UiLaunch,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
# 400+ steps (lead review fix): at ~0.8 s/step a 20-step run finished in
# ~15 s — before the Stop click ever landed. With 420 steps the stop lands
# around step 5-30 and the assertions prove the run halted EARLY (steps_done
# < total), not "ran to completion and lied about it".
STEPS = 420


def _wait_until(pred, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(1.0)
    raise AssertionError(f"timed out waiting for: {what}")


@pytest.fixture()
def ui(tmp_path):
    out_dir = tmp_path / "ui-outputs"
    launch = _UiLaunch(
        tmp_path,
        _base_port(),
        [],
        extra_env={"BACKPROPAGATE_UI__OUTPUT_DIR": str(out_dir)},
    )
    launch.wait_ready()
    launch.token = _banner_token(launch)
    yield launch, tmp_path, out_dir
    launch.stop()


def test_p1_training_flow_via_browser(ui):
    """Drive a real training run from Chrome: start, reload-reattach, stop."""
    launch, tmp_path, sandbox = ui
    base = f"http://127.0.0.1:{launch.port}"
    token_url = f"{base}/?token={launch.token}"

    # Dataset inside the UI sandbox (JobManager refuses anything outside).
    sandbox.mkdir(parents=True, exist_ok=True)
    dataset = sandbox / "p1-flow.jsonl"
    src = REPO_ROOT / "examples" / "quickstart.jsonl"
    assert src.exists(), "examples/quickstart.jsonl missing"
    shutil.copy2(src, dataset)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto(token_url)
        page.wait_for_load_state("networkidle")

        # Fill the form: small model, sandboxed dataset, 20 steps, rank 8.
        page.get_by_label("HuggingFace model id").fill(MODEL_ID)
        page.get_by_label("Path to training dataset (JSONL)").fill(str(dataset))
        page.get_by_label("Number of training steps").fill(str(STEPS))
        page.get_by_label("LoRA rank (r)").fill("8")

        start = page.get_by_role("button", name="Start training")
        start.click()

        # The progress card appears; step counter begins to advance; the
        # config form is LOCKED while the run is live (director fix #9).
        page.wait_for_selector("text=TRAINING", timeout=180_000)
        assert page.get_by_label("HuggingFace model id").is_disabled(), (
            "model field must be disabled while a run is active"
        )

        def step_advanced() -> bool:
            return any(
                r.get("kind") == "step" and int(r.get("step") or 0) >= 3
                for r in _read_all_events(sandbox)
            )

        _wait_until(step_advanced, 600.0, "first step events from the child")

        # Reload mid-run: the banner/pill reattaches from on-disk state.
        page.reload()
        page.wait_for_selector("text=TRAINING", timeout=30_000)

        # Stop and save: cooperative stop writes control.json; the child
        # saves a checkpoint and exits; the banner shows the final state.
        page.get_by_role("button", name="Stop and save checkpoint").click()

        def run_terminal() -> bool:
            rows = _read_all_events(sandbox)
            return any(r.get("kind") == "done" for r in rows)

        _wait_until(run_terminal, 900.0, "terminal done event after stop")

        dones = [r for r in _read_all_events(sandbox) if r.get("kind") == "done"]
        assert dones[-1]["status"] == "stopped"
        # Lead fix #3: a stopped run reports the step it reached (not the
        # requested total).
        assert 0 < dones[-1]["steps_done"] < STEPS, dones[-1]
        # Lead fix #12c: the "saving" phase is emitted once, not per step.
        phases = [
            r for r in _read_all_events(sandbox)
            if r.get("kind") == "phase" and r.get("phase") == "saving"
        ]
        assert len(phases) == 1, f"expected ONE saving phase row, got {len(phases)}"

        # Checkpoint artifacts exist and are loadable-shaped.
        run_dir = _latest_run_dir(sandbox)
        output = run_dir / "output"
        ckpts = list(output.glob("checkpoint-*")) + [output / "lora"]
        adapters = [c for c in ckpts if (c / "adapter_model.safetensors").exists() or (c / "adapter_model.bin").exists()]
        assert adapters, f"no adapter checkpoint under {output}: {ckpts}"

        # run_history entry exists (trainer-owned contract).
        history = output / "run_history.json"
        assert history.exists(), f"run_history.json missing under {output}"
        records = json.loads(history.read_text(encoding="utf-8"))
        runs = records.get("runs") if isinstance(records, dict) else records
        assert runs, "run_history.json has no runs"

        # Job record ended truthfully.
        job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        assert job.get("status") in ("stopped", "done")
        # Lead fix #12a: spawn records carry redacted argv (no drive paths).
        assert not any(str(a).startswith("E:") for a in job.get("argv_tail", []))

        # Lead fix #4: the page says "stopped", and the recovery banner is
        # NOT mis-framed as "Recovered." (this page started the run).
        page.wait_for_selector("text=stopped", timeout=30_000)
        assert not page.get_by_text("Recovered.").count(), (
            "'Recovered.' must only appear when reattaching to a foreign run"
        )

        # Post-run "next steps" panel is up with the saved path shown.
        page.wait_for_selector("text=Saved to:", timeout=30_000)
        browser.close()


def _events_files(sandbox: Path) -> list[Path]:
    return sorted((sandbox / "jobs").glob("run_*/events.jsonl"))


def _read_all_events(sandbox: Path) -> list[dict]:
    rows: list[dict] = []
    for path in _events_files(sandbox):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest_run_dir(sandbox: Path) -> Path:
    candidates = sorted(
        (sandbox / "jobs").glob("run_*"), key=lambda p: p.stat().st_mtime
    )
    assert candidates, f"no job run dirs under {sandbox / 'jobs'}"
    return candidates[-1]
