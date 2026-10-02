# ui-v2 P2 GATE: real-browser, real-GPU multi-run + export through the web UI.
"""End-to-end: Multi-run page -> JobManager -> `backprop multi-run`;
run page -> Export page -> `backprop export --format gguf`.

The P2 acceptance test for ui-v2 (docs/handoff-2026-10-01-ui-v2.md). Heavy:
a real ``backprop ui`` server, the real Chrome browser and real training of
a ~135M model on the local GPU. It runs BY HAND on the dev rig, never in CI,
like tests/test_ui_v2_p1_flow.py.

Gating: ``playwright`` + Chrome, CUDA, and ``BACKPROPAGATE_UI_FLOW=1``.
Optional: ``BACKPROPAGATE_UI_SHOT_DIR=<dir>`` saves a screenshot at each
state (running, stopped, finished, exporting, exported) for review.

The GGUF export needs a llama.cpp converter, as on the CLI: a clone found by
the export probe, or ``BACKPROPAGATE_LLAMA_CPP_PATH`` pointing at its
``convert_hf_to_gguf.py`` (the Store build bundles one).

Run (rig):

    BACKPROPAGATE_UI_FLOW=1 BACKPROPAGATE_LLAMA_CPP_PATH=E:/AI/llama.cpp-src/convert_hf_to_gguf.py \
        PYTHONPATH=. python -m pytest tests/test_ui_v2_p2_flow.py -v -s

Last run on the RTX 5090 (2026-10-02): passed in 4 min 25 s.
"""

from __future__ import annotations

import json
import os
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
_SHOT_DIR = os.environ.get("BACKPROPAGATE_UI_SHOT_DIR", "").strip()


def _shot(page, name: str) -> None:
    if not _SHOT_DIR:
        return
    out = Path(_SHOT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    time.sleep(2.0)
    page.screenshot(path=str(out / f"{name}.png"))


def _wait_until(pred, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(1.0)
    raise AssertionError(f"timed out waiting for: {what}")


def _job_dirs(sandbox: Path) -> list[Path]:
    return sorted((sandbox / "jobs").glob("run_*"), key=lambda p: p.stat().st_mtime)


def _events(job_dir: Path) -> list[dict]:
    rows = []
    path = job_dir / "events.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _terminal(job_dir: Path) -> dict | None:
    for row in reversed(_events(job_dir)):
        if row.get("kind") in ("done", "error"):
            return row
    return None


def _new_job(sandbox: Path, before: set[str]) -> Path:
    found: list[Path] = []

    def appeared() -> bool:
        found[:] = [d for d in _job_dirs(sandbox) if d.name not in before]
        return bool(found)

    _wait_until(appeared, 60.0, "a new job directory")
    return found[-1]


def _run_to_end(job_dir: Path, timeout_s: float = 900.0) -> dict:
    _wait_until(lambda: _terminal(job_dir) is not None, timeout_s, f"{job_dir.name} to finish")
    terminal = _terminal(job_dir)
    assert terminal is not None
    if terminal.get("kind") == "error":
        log = (job_dir / "output.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        raise AssertionError(f"{job_dir.name} failed: {terminal}\n--- log tail ---\n{log}")
    return terminal


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
    yield launch, out_dir
    launch.stop()


def test_p2_multi_run_and_export_via_browser(ui):
    launch, sandbox = ui
    base = f"http://127.0.0.1:{launch.port}"

    # Enough rows for two runs of a few samples each.
    sandbox.mkdir(parents=True, exist_ok=True)
    rows = (REPO_ROOT / "examples" / "quickstart.jsonl").read_text(encoding="utf-8").splitlines()
    dataset = sandbox / "p2-flow.jsonl"
    dataset.write_text("\n".join(rows * 12) + "\n", encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(f"{base}/?token={launch.token}")
        page.wait_for_load_state("networkidle")

        # 1. A short single run, so there is an adapter to export.
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_label("HuggingFace model id").fill(MODEL_ID)
        page.get_by_label("Path to training dataset (JSONL)").fill(str(dataset))
        page.get_by_label("Number of training steps").fill("20")
        page.get_by_label("LoRA rank (r)").fill("8")
        page.get_by_role("button", name="Start training").click()
        sft_dir = _new_job(sandbox, before)
        assert _run_to_end(sft_dir)["status"] == "done"
        sft_job = json.loads((sft_dir / "job.json").read_text(encoding="utf-8"))

        # 2. A 2-run multi-run, left to finish.
        page.goto(base + "/multi-run")
        page.wait_for_load_state("networkidle")
        page.get_by_label("HuggingFace model id").fill(MODEL_ID)
        page.get_by_label("Path to training dataset (JSONL)").fill(str(dataset))
        page.get_by_label("Number of runs in the sweep").fill("2")
        page.get_by_label("Training steps in each run").fill("15")
        page.get_by_label("Training samples in each run").fill("20")
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_role("button", name="Start multi-run").click()
        multi_dir = _new_job(sandbox, before)
        _wait_until(
            lambda: any(r.get("kind") == "step" for r in _events(multi_dir)),
            600.0,
            "the multi-run's first step",
        )
        page.get_by_text("MULTI-RUN").first.wait_for(timeout=30_000)
        _shot(page, "multi-run-running")
        done = _run_to_end(multi_dir)
        assert done["status"] == "done"
        rows_ = _events(multi_dir)
        assert [r["run"] for r in rows_ if r.get("kind") == "run"] == [1, 2]
        steps = [r for r in rows_ if r.get("kind") == "step"]
        assert steps and all(r["total_steps"] == 30 for r in steps)
        assert max(r["step"] for r in steps) > 15, "steps must be session-wide"
        page.get_by_text("Multi-run completed.").first.wait_for(timeout=30_000)
        _shot(page, "multi-run-finished")

        # 3. A multi-run stopped during its first run: the session ends.
        page.get_by_label("Number of runs in the sweep").fill("3")
        page.get_by_label("Training steps in each run").fill("200")
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_role("button", name="Start multi-run").click()
        stop_dir = _new_job(sandbox, before)
        _wait_until(
            lambda: any(r.get("kind") == "step" for r in _events(stop_dir)),
            600.0,
            "the stoppable multi-run's first step",
        )
        page.get_by_role("button", name="Stop and save checkpoint").click()
        _shot(page, "multi-run-stopping")
        stopped = _run_to_end(stop_dir)
        assert stopped["status"] == "stopped"
        assert 0 < stopped["steps_done"] < 600
        assert [r["run"] for r in _events(stop_dir) if r.get("kind") == "run"] == [1], (
            "Stop must end the session, not start the next run"
        )
        page.get_by_text("runs merged so far were kept").first.wait_for(timeout=30_000)
        _shot(page, "multi-run-stopped")

        # 4. Export the single run's model as GGUF, from its run page.
        page.goto(f"{base}/runs/{sft_job['run_id']}")
        page.wait_for_load_state("networkidle")
        page.get_by_role("button", name="Export the model").first.wait_for(timeout=30_000)
        _shot(page, "run-detail")
        page.get_by_role("button", name="Export the model").first.click()
        page.wait_for_url("**/export", timeout=30_000)
        page.wait_for_load_state("networkidle")
        page.get_by_role("radio", name="GGUF").click()
        page.get_by_role("radio", name="Q8_0").click()
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_role("button", name="Export", exact=True).click()
        export_dir = _new_job(sandbox, before)
        page.get_by_text("EXPORTING").first.wait_for(timeout=60_000)
        _shot(page, "export-running")
        exported = _run_to_end(export_dir, timeout_s=1800.0)
        assert exported["status"] == "done"
        out_path = Path(exported["output_path"])
        assert out_path.exists(), out_path
        ggufs = [out_path] if out_path.suffix == ".gguf" else list(out_path.rglob("*.gguf"))
        assert ggufs and ggufs[0].stat().st_size > 10 * 1024 * 1024, ggufs
        page.get_by_text("Export complete").first.wait_for(timeout=30_000)
        _shot(page, "export-finished")

        # 5. Runs lists the UI runs with real statuses.
        page.goto(base + "/runs")
        page.wait_for_load_state("networkidle")
        page.get_by_text("stopped").first.wait_for(timeout=30_000)
        _shot(page, "runs")
        browser.close()
