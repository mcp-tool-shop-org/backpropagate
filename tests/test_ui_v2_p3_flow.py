# ui-v2 P3 GATE: real-browser, real-GPU checks of the new Single run controls.
"""Each new control starts a run with the setting it shows; the inline VRAM
estimate is ``backprop estimate-vram``'s number.

The P3 acceptance test for ui-v2 (docs/handoff-2026-10-01-ui-v2.md). Heavy:
a real ``backprop ui`` server, the real Chrome browser and real training on
the local GPU. It runs BY HAND on the dev rig, never in CI, like the P1 and
P2 flow tests.

1. Preset (Llama 3.2 1B Instruct) + LoRA mode (16-bit base) + the Fast LoRA
   shape + Advanced (run name, temperature limit): the child argv carries
   every flag, and the saved adapter has rank 16 / alpha 32 on q_proj and
   v_proj. The estimate shown before Start equals ``backprop estimate-vram``
   for the same settings, and the run's measured VRAM peak is printed next
   to it.
2. Method ORPO on a preference dataset (SmolLM2-135M, QLoRA): the child runs
   ``--method orpo`` with the beta from the form and finishes.
3. Full fine-tune (SmolLM2-135M, SFT): ``--mode full``, finishes, saves full
   weights rather than an adapter.

Gating: ``playwright`` + Chrome, CUDA, and ``BACKPROPAGATE_UI_FLOW=1``.
Optional: ``BACKPROPAGATE_UI_SHOT_DIR=<dir>`` saves screenshots.

Run (rig):

    BACKPROPAGATE_UI_FLOW=1 PYTHONPATH=. python -m pytest tests/test_ui_v2_p3_flow.py -v -s
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

SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"
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


def _run_to_end(job_dir: Path, timeout_s: float = 1200.0) -> dict:
    _wait_until(lambda: _terminal(job_dir) is not None, timeout_s, f"{job_dir.name} to finish")
    terminal = _terminal(job_dir)
    assert terminal is not None
    if terminal.get("kind") == "error" or terminal.get("status") not in ("done",):
        log = (job_dir / "output.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        raise AssertionError(f"{job_dir.name} ended {terminal}\n--- log tail ---\n{log}")
    return terminal


def _argv(job_dir: Path) -> list[str]:
    return list(json.loads((job_dir / "job.json").read_text(encoding="utf-8"))["argv_tail"])


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _cli_estimate(*args: str) -> float:
    proc = subprocess.run(
        [sys.executable, "-m", "backpropagate", "estimate-vram", *args, "--json"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    out = proc.stdout
    decoder, i = json.JSONDecoder(), 0
    while (i := out.find("{", i)) != -1:
        try:
            obj, _end = decoder.raw_decode(out, i)
            if isinstance(obj, dict) and obj.get("per_config_estimate"):
                return float(obj["per_config_estimate"]["total_gb"])
        except json.JSONDecodeError:
            pass
        i += 1
    raise AssertionError(f"no estimate in:\n{out}\n{proc.stderr[-2000:]}")


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


def _fill_common(page, dataset: Path, steps: str, batch: str) -> None:
    page.get_by_label("Path to training dataset (JSONL)").fill(str(dataset))
    page.get_by_label("Number of training steps").fill(steps)
    page.get_by_label("Batch size (number or auto)").fill(batch)


@pytest.mark.timeout(3600)
def test_p3_new_controls_reach_the_run(ui):
    launch, sandbox = ui
    base = f"http://127.0.0.1:{launch.port}"
    sandbox.mkdir(parents=True, exist_ok=True)
    rows = (REPO_ROOT / "examples" / "quickstart.jsonl").read_text(encoding="utf-8").splitlines()
    sft_data = sandbox / "p3-sft.jsonl"
    sft_data.write_text("\n".join(rows * 12) + "\n", encoding="utf-8")
    pref_data = sandbox / "p3-pref.jsonl"
    pref_rows = [
        {"prompt": f"Say something kind about the number {i}.",
         "chosen": f"{i} is a wonderful number with a lot to offer.",
         "rejected": f"{i} is boring."}
        for i in range(48)
    ]
    pref_data.write_text("\n".join(json.dumps(r) for r in pref_rows) + "\n", encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(f"{base}/?token={launch.token}")
        page.wait_for_load_state("networkidle")

        # ---- 1. preset + LoRA (16-bit) + Fast shape + Advanced -------------
        page.get_by_label("Model preset").click()
        page.get_by_role("option", name="Llama 3.2 1B Instruct").click()
        page.get_by_role("radio", name="LoRA", exact=True).click()
        page.get_by_role("button", name="Fast").click()
        _fill_common(page, sft_data, "30", "2")
        page.get_by_text("Advanced", exact=True).click()
        page.get_by_label("Run name for experiment trackers (optional)").fill("p3-flow")
        page.get_by_label("GPU temperature limit in Celsius (stop and save above this)").fill("95")
        cli_total = _cli_estimate(
            "meta-llama/Llama-3.2-1B-Instruct", "--lora-r", "16", "--batch-size", "2",
            "--target-modules", "q_proj,v_proj",
            "--no-4bit",
        )

        # The estimate refreshes off the event loop as each field changes;
        # the shown number must settle on the CLI's for the same settings.
        def _ui_total() -> float | None:
            text = page.locator("#bp-vram-estimate").inner_text().strip()
            try:
                return float(text.split(" GB")[0]) if text else None
            except ValueError:
                return None

        try:
            _wait_until(
                lambda: (t := _ui_total()) is not None and abs(t - cli_total) <= 0.05,
                90.0,
                "the inline estimate to match estimate-vram",
            )
        except AssertionError:
            raise AssertionError(
                (page.locator("#bp-vram-estimate").inner_text(), cli_total)
            ) from None
        shown = page.locator("#bp-vram-estimate").inner_text()
        verdict = page.locator("#bp-vram-verdict").inner_text()
        _shot(page, "train-configured")
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_role("button", name="Start training").click()
        job = _new_job(sandbox, before)
        done = _run_to_end(job)
        argv = _argv(job)
        assert _flag(argv, "--model") == "meta-llama/Llama-3.2-1B-Instruct"
        assert "--no-4bit" in argv
        assert (_flag(argv, "--lora-r"), _flag(argv, "--lora-alpha")) == ("16", "32")
        assert _flag(argv, "--target-modules") == "q_proj,v_proj"
        assert _flag(argv, "--batch-size") == "2"
        assert _flag(argv, "--run-name") == "p3-flow"
        assert _flag(argv, "--gpu-max-temp") == "95"
        adapter_cfg = next(Path(done["output_path"]).rglob("adapter_config.json"))
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        assert (cfg["r"], cfg["lora_alpha"]) == (16, 32)
        assert set(cfg["target_modules"]) == {"q_proj", "v_proj"}
        peak = max(
            float(r.get("vram_reserved_gib") or 0.0)
            for r in _events(job) if r.get("kind") == "step"
        )
        print(f"\nP3 estimate: UI {shown} ({verdict}), CLI {cli_total:.2f} GB; "
              f"measured peak reserved {peak:.2f} GiB")
        page.get_by_text("Run completed.").first.wait_for(timeout=30_000)
        _shot(page, "train-finished")

        # ---- 2. ORPO on preference pairs (QLoRA) ----------------------------
        page.goto(base + "/")
        page.wait_for_load_state("networkidle")
        page.get_by_label("HuggingFace model id").fill(SMALL)
        page.get_by_role("radio", name="QLoRA").click()
        page.get_by_role("radio", name="ORPO").click()
        page.get_by_label("ORPO beta").fill("0.2")
        _fill_common(page, pref_data, "12", "2")
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_role("button", name="Start training").click()
        job = _new_job(sandbox, before)
        _run_to_end(job)
        argv = _argv(job)
        assert _flag(argv, "--method") == "orpo"
        assert _flag(argv, "--orpo-beta") == "0.2"
        assert "--no-4bit" not in argv

        # ---- 3. Full fine-tune (SFT) ----------------------------------------
        page.goto(base + "/")
        page.wait_for_load_state("networkidle")
        page.get_by_label("HuggingFace model id").fill(SMALL)
        page.get_by_role("radio", name="SFT").click()
        page.get_by_role("radio", name="Full fine-tune").click()
        page.get_by_text("Full fine-tuning trains every weight").first.wait_for(timeout=10_000)
        _fill_common(page, sft_data, "12", "2")
        before = {d.name for d in _job_dirs(sandbox)}
        page.get_by_role("button", name="Start training").click()
        job = _new_job(sandbox, before)
        done = _run_to_end(job)
        argv = _argv(job)
        assert _flag(argv, "--mode") == "full"
        assert "--lora-alpha" not in argv and "--target-modules" not in argv
        out = Path(done["output_path"])
        assert list(out.rglob("model*.safetensors")), sorted(p.name for p in out.rglob("*"))
        assert not list(out.rglob("adapter_config.json"))
        _shot(page, "train-full-finished")
        browser.close()
