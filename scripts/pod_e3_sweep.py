#!/usr/bin/env python3
"""E3 VRAM sweep: measure the QLoRA presets' real GPU peak, point by point.

For every QLoRA preset in ``MODEL_PRESETS`` (measured SMALL TO LARGE, so the HF
cache, which must live on the container disk, only ever holds one preset plus
the next one being prefetched), at batch 1, 2, 4 and 6, at the preset's own
default sequence window, with Unsloth off and on: load the model in a fresh
process, train a handful of steps and write ONE JSON receipt for the point.

Why a process per point: an OOM, a CUDA error or a leaked paged-optimizer
buffer in one point must not touch the next, and the NVML peak of a point must
not include a previous point's cached memory.

What a receipt records (all of it per point; see ``measure_point``):
  * the NVML device peak, sampled at 100 Hz on a thread for the whole process
    (PyTorch's counters do NOT see bitsandbytes' paged optimizer state; handoff
    section 3.2), next to torch max allocated / reserved, the paged-state size,
    the CUDA context, and a per-step trace of the running peaks so a plateau
    can be checked;
  * ``peak_gib_for_fit``: the figure the refit tool fits and the gate judges:
    the NVML sampled peak, or ``torch reserved + context + paged state`` when
    that is larger (a 100 Hz sampler can miss a short transient);
  * seconds per step, load seconds;
  * what ``estimate_vram`` predicts for this exact point: ``predicted_default``
    (the way ``Trainer.estimate_vram()`` and the CLI call it: 7B-class default
    dimensions, name-derived parameter count) and ``predicted_arch`` (the same
    estimator with the model's real dimensions, read from its config);
  * what ``_detect_batch_size`` would resolve on this card;
  * whether ``oom_recovery`` fired. It is OFF here (``oom_recovery=False``), so
    an out-of-memory is recorded as the result for that point, never silently
    halved; ``oom_retries`` is recorded and must be 0 or absent.

Subcommands (see ``--help`` of each)::

    plan       the grid, the cost estimate and what the budget guard drops
    arch       architecture + text-only parameter count per preset (config only)
    point      measure ONE point (what ``run`` spawns)
    run        the orchestrator: plan, prefetch, measure, guard, purge, summarize
    summarize  rebuild summary.json from the receipts

Dry run (no GPU, no network, no weights): add ``--synthetic`` to ``run`` or
``point``. A tiny random-initialised model and a tiny tokenizer are built
locally per preset (``CUDA_VISIBLE_DEVICES=-1``, ``HF_HUB_OFFLINE=1``) so the
whole chain (planner, orchestrator, Trainer, receipts, summary, refit) can be
smoked on CPU before a pod is paid for. Synthetic receipts say so and carry an
RSS stand-in for the peak; the refit refuses them unless ``--allow-dry-run``.

Budget guard. ``plan_budget`` (e3_lib) fits the grid into ``--budget-usd`` before
the run by dropping points in ``drop_order``: highest tier number first (small
presets at batch 2/4, then mid, then large at 2/4, then small and mid at
batch 1/6), Unsloth-on before Unsloth-off, later position first. The 14B / 24B /
32B points at batch 1 and 6 (tier 0, the latent auto-batch question) are
dropped last of all. ``run`` re-applies the same order at run time when actual
timings overrun the estimates, and ``--stop-after-epoch`` / ``E3_STOP_AFTER_EPOCH``
is a hard wall.

Secrets. ``HF_TOKEN`` is read from the environment by the Hub client and is
never written, printed or put on a command line; every receipt and log is passed
through ``scrub`` before it is written, and receipts record only
``hf_token_present: true/false``.

Standards compliance: see the docstring of ``e3_lib.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# The repo this script lives in must win over any other editable install of
# `backpropagate` (the rig's venv is an editable install of the MAIN checkout).
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
import e3_lib as L  # noqa: E402

GIB = L.GIB
RUNS = "runs"


# --------------------------------------------------------------------- args
def _csv(s: str | None) -> list[str] | None:
    return None if not s else [x.strip() for x in s.split(",") if x.strip()]


def _arms(s: str) -> tuple[bool, ...]:
    return {"off": (False,), "on": (True,), "both": (False, True)}[s]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, help="receipt directory (runs/, arch/, plan.json ...)")
        p.add_argument("--presets", default=None, help="comma list (default: every preset)")
        p.add_argument("--batches", default=None, help=f"comma list (default: {','.join(map(str, L.BATCHES))})")
        p.add_argument("--unsloth", choices=["off", "on", "both"], default="both")
        p.add_argument("--steps", type=int, default=L.DEFAULT_STEPS)
        p.add_argument("--synthetic", "--dry-run", dest="synthetic", action="store_true",
                       help="CPU dry run with tiny local models: no GPU, no network, no weights")

    def budget(p: argparse.ArgumentParser) -> None:
        p.add_argument("--budget-usd", type=float,
                       default=float(os.environ.get("E3_BUDGET_USD", L.DEFAULT_BUDGET_USD)))
        p.add_argument("--usd-per-hour", type=float,
                       default=float(os.environ.get("E3_USD_PER_HOUR", L.DEFAULT_USD_PER_HOUR)))
        p.add_argument("--dl-mbps", type=float, default=float(os.environ.get("E3_DL_MBPS", L.DEFAULT_DL_MBPS)))
        p.add_argument("--no-prefetch", action="store_true")

    p = sub.add_parser("plan", help="print the grid, the cost estimate and the drop list")
    common(p), budget(p)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("arch", help="architecture + text-only param count per preset, from config only")
    common(p)

    p = sub.add_parser("point", help="measure one point (spawned by run)")
    common(p)
    p.add_argument("--preset", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)

    p = sub.add_parser("run", help="orchestrate the sweep")
    common(p), budget(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-purge", action="store_true", help="keep weights in the HF cache after a preset")
    p.add_argument("--no-monotone-skip", action="store_true",
                   help="measure every point even when a smaller batch already OOMed")
    p.add_argument("--no-guard", action="store_true", help="ignore the budget (run the whole grid)")
    p.add_argument("--stop-after-epoch", type=int, default=int(os.environ.get("E3_STOP_AFTER_EPOCH", "0") or 0),
                   help="hard wall: unix time after which no new point starts")
    p.add_argument("--allow-missing-unsloth", action="store_true",
                   help="drop Unsloth-on points instead of halting when unsloth is not installed")

    p = sub.add_parser("summarize", help="rebuild summary.json from the receipts")
    p.add_argument("--out", required=True)
    return ap


def select_grid(a: argparse.Namespace) -> tuple[list[L.Preset], list[L.Point]]:
    presets = L.load_presets()
    names = _csv(a.presets)
    if names:
        unknown = sorted(set(names) - {p.name for p in presets})
        if unknown:
            raise SystemExit(f"unknown preset(s): {', '.join(unknown)}")
        presets = [p for p in presets if p.name in names]
    batches = tuple(int(b) for b in _csv(a.batches)) if a.batches else L.BATCHES
    return presets, L.build_points(presets, batches, _arms(a.unsloth))


# --------------------------------------------------------------------- plan
def plan_report(a: argparse.Namespace) -> dict[str, Any]:
    presets, points = select_grid(a)
    plan = L.plan_budget(presets, points, a.budget_usd, a.usd_per_hour, a.steps, a.dl_mbps,
                         not a.no_prefetch)
    full_total, full_per, full_dl = L.estimate_total_seconds(presets, points, a.steps, a.dl_mbps,
                                                             not a.no_prefetch)
    by_tier: dict[int, dict[str, int]] = {}
    for pt in points:
        by_tier.setdefault(pt.tier, {"points": 0, "kept": 0})["points"] += 1
    for pt in plan.keep:
        by_tier[pt.tier]["kept"] += 1
    return {
        "steps_per_point": a.steps,
        "presets_small_to_large": [p.name for p in presets],
        "points_total": len(points),
        "full_grid": {"est_hours": round(full_total / 3600, 2),
                      "est_usd": round(full_total / 3600 * a.usd_per_hour, 2),
                      "exposed_download_min": round(full_dl / 60, 1)},
        "budget": {"usd": a.budget_usd, "usd_per_hour": a.usd_per_hour, "seconds": round(plan.budget_seconds)},
        "kept": len(plan.keep), "dropped": len(plan.dropped),
        "kept_est_hours": round(plan.est_seconds / 3600, 2), "kept_est_usd": round(plan.est_usd, 2),
        "by_tier": {str(k): v for k, v in sorted(by_tier.items())},
        "tier_meaning": {"0": "14B/24B/32B at batch 1 and 6", "1": "7B-13B at batch 1 and 6",
                         "2": "under 7B at batch 1 and 6", "3": "14B+ at batch 2 and 4",
                         "4": "7B-13B at batch 2 and 4", "5": "under 7B at batch 2 and 4"},
        "drop_order_head": [p.tag for p in L.drop_order(points)[:8]],
        "dropped_tags": [p.tag for p in plan.dropped],
        "kept_tags": [p.tag for p in plan.keep],
        "notes": plan.notes,
    }


def cmd_plan(a: argparse.Namespace) -> int:
    rep = plan_report(a)
    if a.json:
        print(json.dumps(rep, indent=1))
        return 0
    print(f"E3 sweep plan: {rep['points_total']} points, {a.steps} steps each; presets small to large:")
    print("  " + ", ".join(rep["presets_small_to_large"]))
    fg = rep["full_grid"]
    print(f"full grid: ~{fg['est_hours']} h, ~${fg['est_usd']} at ${a.usd_per_hour}/h "
          f"(exposed download ~{fg['exposed_download_min']} min with prefetch)")
    print(f"budget: ${a.budget_usd} = {rep['budget']['seconds'] // 60} min -> keep {rep['kept']} points "
          f"(~{rep['kept_est_hours']} h, ~${rep['kept_est_usd']}), drop {rep['dropped']}")
    for t, d in rep["by_tier"].items():
        print(f"  tier {t} ({rep['tier_meaning'][t]}): kept {d['kept']} of {d['points']}")
    print("drop order starts: " + ", ".join(rep["drop_order_head"]))
    return 0


# --------------------------------------------------------------------- arch
def arch_path(out: str, preset: str, synthetic: bool = False) -> str:
    return os.path.join(out, "arch_synthetic" if synthetic else "arch", f"{preset}.json")


def ensure_arch(out: str, preset: L.Preset, model_id: str | None = None) -> dict[str, Any]:
    """Cached architecture of a preset; ``model_id`` overrides the repo (synthetic dry runs: a local dir)."""
    path = arch_path(out, preset.name, synthetic=model_id is not None)
    if os.path.exists(path):
        return L.read_json(path)
    rec = L.fetch_arch(model_id or preset.model_id)
    rec["preset"] = preset.name
    L.write_json(path, rec)
    return rec


def cmd_arch(a: argparse.Namespace) -> int:
    if a.synthetic:
        print("arch: nothing to fetch in a synthetic dry run (each point builds its own tiny model)")
        return 0
    presets, _ = select_grid(a)
    for p in presets:
        try:
            rec = ensure_arch(a.out, p)
            print(f"{p.name}: {rec['hidden_size']}x{rec['num_hidden_layers']} heads={rec['num_attention_heads']} "
                  f"vocab={rec['vocab_size']} text_params={rec['text_params_b']:.3f}B class={rec['loaded_class']}")
        except Exception as exc:  # noqa: BLE001 - one gated/unreachable repo must not stop the rest
            print(f"{p.name}: FAILED {type(exc).__name__}: {L.scrub(str(exc)[:200], L.hf_secrets())}")
    return 0


# ------------------------------------------------------------- measurement
class NvmlSampler:
    """System-wide device memory (NVML), sampled at 100 Hz on a thread.

    torch's allocator counters miss memory allocated outside it, notably
    bitsandbytes' paged optimizer state (CUDA managed memory). NVML sees
    everything resident on the device, including the CUDA context. Same idea as
    ``pod_block_engine.NvmlSampler`` (that module is a CLI and cannot be
    imported); ``mark(phase)`` tracks a named phase's peak separately.
    """

    def __init__(self, hz: float = 100.0) -> None:
        self.ok = False
        self.peak = 0
        self.phase = "start"
        self.phase_peak: dict[str, int] = {}
        self.baseline: int | None = None
        self.error: str | None = None
        self.driver: str | None = None
        self._period = 1.0 / hz
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            drv = pynvml.nvmlSystemGetDriverVersion()
            self.driver = drv.decode() if isinstance(drv, bytes) else drv
            self.baseline = self.used()
            self.ok = True
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def used(self) -> int:
        return int(self._nv.nvmlDeviceGetMemoryInfo(self._h).used)

    def total(self) -> int:
        return int(self._nv.nvmlDeviceGetMemoryInfo(self._h).total)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                u = self.used()
            except Exception:  # noqa: BLE001  # nosec B112 - sampling is best effort
                continue
            self.peak = max(self.peak, u)
            self.phase_peak[self.phase] = max(self.phase_peak.get(self.phase, 0), u)
            self._stop.wait(self._period)

    def mark(self, phase: str) -> None:
        self.phase = phase

    def stop(self) -> None:
        if self.ok:
            self._stop.set()

    def report(self) -> dict[str, Any]:
        if not self.ok:
            return {"nvml": "unavailable", "nvml_error": self.error}
        return {"nvml_peak_gib": round(self.peak / GIB, 3),
                "nvml_baseline_gib": round((self.baseline or 0) / GIB, 3),
                "nvml_phase_peak_gib": {k: round(v / GIB, 3) for k, v in self.phase_peak.items()},
                "nvml_sample_hz": round(1.0 / self._period)}


def package_versions() -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for p in ("torch", "transformers", "trl", "accelerate", "peft", "bitsandbytes", "nvidia-ml-py",
              "unsloth", "unsloth_zoo", "datasets", "backpropagate"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def git_sha() -> str:
    sha = os.environ.get("GIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "-C", os.path.dirname(HERE), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=False).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


OOM_MARKERS = ("out of memory", "OutOfMemory", "RUNTIME_GPU_OOM", "RUNTIME_OOM", "CUBLAS_STATUS_ALLOC_FAILED",
               "cublas_status_alloc_failed")


def is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {getattr(exc, 'code', '')}: {exc}"
    cause = repr(exc.__cause__) if exc.__cause__ is not None else ""
    return any(m in text + cause for m in OOM_MARKERS)


def synthetic_rows(n_rows: int, window: int) -> list[dict]:
    """Deterministic chat rows each LONGER than ``window`` tokens (~2 x window of text).

    Same construction as tests/test_qlora_presets_smoke.py: with packing off an
    over-length row is truncated to exactly ``window`` tokens, so every training
    example, whichever the sampler picks, has the preset's shipped shape and a
    step input is (batch, window).
    """
    topics = ["binary search", "hash tables", "TCP handshakes", "garbage collection",
              "B-trees", "unicode normalization", "rate limiting", "consistent hashing"]
    n_sentences = (2 * window) // 25
    rows = []
    for i in range(n_rows):
        topic = topics[i % len(topics)]
        answer = " ".join(
            f"Point {j + 1} about {topic}: it trades memory for time in case {i}-{j}, "
            f"and the invariant that makes it correct must hold after every update."
            for j in range(n_sentences))
        rows.append({"messages": [
            {"role": "user", "content": f"Explain {topic} in depth (variant {i})."},
            {"role": "assistant", "content": answer}]})
    return rows


def build_synthetic_model(root: str, preset_index: int, rows: list[dict]) -> str:
    """A tiny random-initialised Llama + a tiny BPE tokenizer, built locally. No network."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = os.path.join(root, f"synthetic_model_{preset_index}")
    if os.path.exists(os.path.join(d, "config.json")):
        return d
    os.makedirs(d, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [m["content"] for r in rows for m in r["messages"]] + ["<|im_start|>user\n", "<|im_end|>\n", "<|im_start|>assistant\n"]
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                                   eos_token="</s>", pad_token="<pad>")
    fast.save_pretrained(d)
    torch.manual_seed(preset_index)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=64 + 16 * (preset_index % 3), intermediate_size=128,
                      num_hidden_layers=2 + (preset_index % 2), num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, bos_token_id=1, eos_token_id=2, pad_token_id=3,
                      tie_word_embeddings=False)
    LlamaForCausalLM(cfg).save_pretrained(d)
    return d


def describe_optimizer(t: Any, torch: Any) -> dict[str, Any]:
    """Which optimizer ran and the size of its state, paged (managed-memory) part separately."""
    out: dict[str, Any] = {}
    tr = getattr(t, "_trainer", None)
    opt = getattr(tr, "optimizer", None)
    inner = getattr(opt, "optimizer", opt)
    if inner is None:
        return {"optimizer_class": None, "optimizer_state_gib": None, "optimizer_paged_state_gib": None}
    out["optimizer_class"] = f"{type(inner).__module__}.{type(inner).__name__}"
    paged = total = 0
    try:
        for st in inner.state.values():
            for v in (st.values() if isinstance(st, dict) else []):
                if torch.is_tensor(v):
                    b = v.numel() * v.element_size()
                    total += b
                    if getattr(v, "is_paged", False):
                        paged += b
    except Exception as exc:  # noqa: BLE001 - recorded
        out["optimizer_state_error"] = f"{type(exc).__name__}: {exc}"[:200]
    out["optimizer_state_gib"] = round(total / GIB, 3)
    out["optimizer_paged_state_gib"] = round(paged / GIB, 3)
    return out


def measure_point(a: argparse.Namespace) -> dict[str, Any]:
    """Measure one point in this process and return its receipt (never raises)."""
    unsloth_on = a.unsloth == "on"
    presets = {p.name: p for p in L.load_presets()}
    if a.preset not in presets:
        raise SystemExit(f"unknown preset {a.preset!r}")
    preset = presets[a.preset]
    secrets = L.hf_secrets()
    tag = a.tag or L.point_tag(preset.name, a.batch, unsloth_on)
    t_proc = time.perf_counter()

    os.environ.setdefault("UNSLOTH_AUTO_INSTALL", "0")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["BACKPROPAGATE_TRAINING__SEED"] = str(a.seed)
    os.environ["BACKPROPAGATE_TRAINING__LOGGING_STEPS"] = "1"
    os.environ["BACKPROPAGATE_TRAINING__SAVE_STEPS"] = "1000000"

    import torch

    torch.manual_seed(a.seed)
    cuda = torch.cuda.is_available() and not a.synthetic
    total_vram = torch.cuda.get_device_properties(0).total_memory if cuda else 0
    card = torch.cuda.get_device_name(0) if cuda else "cpu (dry run)"

    nvml = NvmlSampler() if cuda else None
    baseline_before_ctx = nvml.used() if nvml and nvml.ok else None
    ctx_gib = None
    if cuda:
        torch.zeros(1, device="cuda")  # create the CUDA context so it can be measured
        torch.cuda.synchronize()
        if nvml and nvml.ok and baseline_before_ctx is not None:
            ctx_gib = round((nvml.used() - baseline_before_ctx) / GIB, 3)

    def peaks() -> tuple[float, float]:
        return ((torch.cuda.max_memory_allocated() / GIB, torch.cuda.max_memory_reserved() / GIB)
                if cuda else (0.0, 0.0))

    def mark(phase: str) -> None:
        if nvml and nvml.ok:
            nvml.mark(phase)

    workdir = os.path.join(a.out, "work", tag)
    os.makedirs(workdir, exist_ok=True)

    window = min(preset.window, 128) if a.synthetic else preset.window
    model_id = preset.model_id
    n_rows = max(8, 2 * a.steps * a.batch)
    rows = synthetic_rows(n_rows, window)
    data_path = os.path.join(workdir, "sft.jsonl")
    with open(data_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if a.synthetic:
        model_id = build_synthetic_model(os.path.join(a.out, "work"), list(presets).index(preset.name), rows)

    rec: dict[str, Any] = {
        "mode": "e3_point", "tag": tag, "preset": preset.name, "model": preset.model_id,
        "batch": a.batch, "unsloth_requested": unsloth_on, "window": window, "preset_window": preset.window,
        "lora_r": preset.lora_r, "packing": False, "steps": a.steps, "seed": a.seed,
        "dry_run": bool(a.synthetic), "oom_recovery": False,
        "manifest": {
            "git_sha": git_sha(), "image": os.environ.get("POD_IMAGE", "unknown"), "gpu": card,
            "vram_total_gib": round(total_vram / GIB, 2), "cuda_runtime": torch.version.cuda,
            "cuda_driver": getattr(nvml, "driver", None), "versions": package_versions(),
            "hf_token_present": bool(secrets), "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "hf_home": os.environ.get("HF_HOME"), "dry_run": bool(a.synthetic),
        },
        "card": card, "card_total_gib": round(total_vram / GIB, 3),
        "cuda_context_gib": ctx_gib,
    }
    if a.synthetic:
        rec["window_note"] = f"synthetic dry run: window capped at {window} (preset window {preset.window})"

    steps_t: list[float] = []
    losses: list[float] = []
    nvml_by_step: list[float] = []
    reserved_by_step: list[float] = []
    alloc_by_step: list[float] = []
    step_shapes: list[list[int]] = []
    phase = "init"
    t0 = time.perf_counter()
    peak_rss_gib = 0.0

    def rss_gib() -> float:
        try:
            import psutil

            return psutil.Process().memory_info().rss / GIB
        except Exception:  # noqa: BLE001
            return 0.0

    try:
        arch = ensure_arch(a.out, preset, model_id if a.synthetic else None)
        rec["arch"] = arch

        from backpropagate.trainer import Trainer, TrainingCallback, estimate_vram

        kwargs: dict[str, Any] = {
            "model": model_id, "use_unsloth": unsloth_on, "unsloth_fallback": False, "mode": "lora",
            "lora_r": preset.lora_r, "lora_alpha": preset.lora_r, "max_seq_length": window, "packing": False,
            "batch_size": a.batch, "gradient_accumulation": 1, "oom_recovery": False,
            # loss masking does not change memory; False keeps Unsloth's response-marker detection
            # (which can fail on an exotic chat template) from voiding a whole arm
            "train_on_responses": False,
            "output_dir": os.path.join(workdir, "out"), "report_to": "none"}
        if a.synthetic:
            kwargs.update(load_in_4bit=False, use_unsloth=False)
            rec["synthetic_note"] = "load_in_4bit=False and use_unsloth=False: bitsandbytes/Unsloth need CUDA"
        rec["trainer_kwargs"] = {k: v for k, v in kwargs.items() if k not in ("output_dir", "model")}

        from backpropagate.config import settings as bp_settings

        bp_settings.training.seed = a.seed
        bp_settings.training.logging_steps = 1
        bp_settings.training.save_steps = 1_000_000

        # No trainer checkpoints: disk, not evidence (a 32B adapter + optimizer state is large).
        orig_build = Trainer._build_training_args

        def _build_no_ckpt(self, **kw):  # type: ignore[no-untyped-def]
            cfg = orig_build(self, **kw)
            from transformers.trainer_utils import SaveStrategy

            cfg.save_strategy = SaveStrategy.NO
            return cfg

        Trainer._build_training_args = _build_no_ckpt  # type: ignore[method-assign]

        import trl

        orig_step = trl.SFTTrainer.training_step

        def _spy_step(self, model, inputs, *args, **kw):  # type: ignore[no-untyped-def]
            ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if ids is not None and len(step_shapes) < 3:
                step_shapes.append([int(x) for x in ids.shape])
            return orig_step(self, model, inputs, *args, **kw)

        trl.SFTTrainer.training_step = _spy_step  # type: ignore[method-assign]

        t = Trainer(**kwargs)
        rec["unsloth_active"] = bool(t.use_unsloth)
        if unsloth_on and not t.use_unsloth and not a.synthetic:
            rec["status"] = "skipped_unsloth_unavailable"
            rec["error"] = "use_unsloth=True was requested but the Trainer resolved use_unsloth=False (extra not installed)"
            return rec
        rec["optim_setting"] = t.optim
        rec["lr_used"] = t.learning_rate
        rec["max_seq_length_used"] = t.max_seq_length
        rec["batch_resolved"] = t.batch_size

        # What the library would do / say for this point.
        rec["auto_batch_choice"] = int(t._detect_batch_size())
        if a.synthetic:  # the local model dir has no meaningful name; call the estimator for the real id
            default_est = estimate_vram(model=preset.model_id, mode="lora", lora_r=preset.lora_r,
                                        batch_size=a.batch, max_seq_length=window, use_unsloth=bool(t.use_unsloth))
        else:
            default_est = t.estimate_vram()  # exactly what the library says to an operator
        rec["predicted_default"] = dataclasses.asdict(default_est)
        pa = arch or {}
        rec["predicted_arch"] = dataclasses.asdict(estimate_vram(
            model=preset.model_id, mode="lora", lora_r=preset.lora_r, batch_size=a.batch,
            max_seq_length=window, hidden_dim=pa.get("hidden_size", 4096),
            num_layers=pa.get("num_hidden_layers", 32), num_heads=pa.get("num_attention_heads", 32),
            vocab_size=pa.get("vocab_size", 152064), param_count_billions=pa.get("text_params_b"),
            use_unsloth=bool(t.use_unsloth)))

        phase = "load"
        mark("load")
        t.load_model()
        rec["load_s"] = round(time.perf_counter() - t0, 1)
        try:
            rec["trainable_params"] = sum(p.numel() for p in t._model.parameters() if p.requires_grad)
            rec["model_class"] = type(t._model).__name__
            # Does the loader that ran instantiate a vision tower? (Qwen3.5-4B is tagged image-text-to-text.)
            vis = [(n, p.numel()) for n, p in t._model.named_parameters() if "visual" in n or "vision" in n]
            rec["vision_tensors_loaded"] = len(vis)
            rec["vision_params_loaded"] = sum(c for _, c in vis)
            rec["lora_adapted_modules"] = getattr(t, "lora_adapted_modules", None)
        except Exception as exc:  # noqa: BLE001
            rec["model_introspection_error"] = f"{type(exc).__name__}: {exc}"[:200]
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        rec["nvml_after_load_gib"] = round(nvml.used() / GIB, 3) if nvml and nvml.ok else None

        def on_step(step, loss):  # type: ignore[no-untyped-def]  # noqa: ARG001
            nonlocal peak_rss_gib
            if len(steps_t) >= a.steps:  # the trainer logs once more at the end (run summary): not a step
                return
            steps_t.append(time.perf_counter())
            losses.append(round(float(loss), 4))
            if nvml and nvml.ok:
                nvml_by_step.append(round(nvml.peak / GIB, 3))
            al, rs = peaks()
            alloc_by_step.append(round(al, 3))
            reserved_by_step.append(round(rs, 3))
            peak_rss_gib = max(peak_rss_gib, rss_gib())

        phase = "train"
        mark("train")
        t1 = time.perf_counter()
        run = t.train(data_path, steps=a.steps, samples=n_rows, callback=TrainingCallback(on_step=on_step))
        mark("after_train")
        rec["train_s"] = round(time.perf_counter() - t1, 1)
        rec["oom_retries"] = run.metadata.get("oom_retries")
        rec["effective_batch_size"] = run.metadata.get("effective_batch_size")
        rec["final_loss"] = run.final_loss
        rec.update(describe_optimizer(t, torch))
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - an OOM at a point is a result, not a crash
        msg = f"{type(exc).__name__}: {getattr(exc, 'code', '')}: {exc}"
        rec["status"] = "oom" if is_oom(exc) else "error"
        rec["oom_phase"] = phase if rec["status"] == "oom" else None
        rec["error"] = L.scrub(msg[:2000], secrets)
        rec["traceback_tail"] = L.scrub(traceback.format_exc()[-3000:], secrets)
    finally:
        al, rs = peaks()
        rec["torch_max_allocated_gib"] = round(al, 3)
        rec["torch_max_reserved_gib"] = round(rs, 3)
        rec["torch_max_allocated_by_step_gib"] = alloc_by_step
        rec["torch_max_reserved_by_step_gib"] = reserved_by_step
        rec["nvml_peak_by_step_gib"] = nvml_by_step
        rec["losses"] = losses
        rec["steps_completed"] = len(losses)
        gaps = [b - a_ for a_, b in zip(steps_t, steps_t[1:])]
        rec["s_per_step"] = round(statistics.median(gaps), 4) if gaps else None
        rec["s_per_step_source"] = "median gap between per-step log callbacks"
        rec["step_gaps_s"] = [round(g, 3) for g in gaps]
        rec["step_input_shapes"] = step_shapes
        rec["shape_ok"] = bool(step_shapes) and all(s == [a.batch, window] for s in step_shapes)
        if len(nvml_by_step) >= 5:
            rec["plateau_ok"] = nvml_by_step[-1] <= 1.01 * nvml_by_step[3]
        elif len(reserved_by_step) >= 5:
            rec["plateau_ok"] = reserved_by_step[-1] <= 1.01 * reserved_by_step[3]
        else:
            rec["plateau_ok"] = None
        if nvml is not None:
            rec.update(nvml.report())
            nvml.stop()
        paged = rec.get("optimizer_paged_state_gib") or 0.0
        if cuda and rec.get("nvml_peak_gib") is not None:
            composite = round(rs + (ctx_gib or 0.0) + paged, 3)
            rec["peak_composite_gib"] = composite
            if composite > rec["nvml_peak_gib"]:
                rec["peak_gib_for_fit"], rec["peak_source"] = composite, "torch_reserved+context+paged"
            else:
                rec["peak_gib_for_fit"], rec["peak_source"] = rec["nvml_peak_gib"], "nvml_sampled"
        else:
            rec["peak_gib_for_fit"] = round(peak_rss_gib, 3)
            rec["peak_source"] = "cpu_rss (dry run; not a VRAM figure)"
        rec["wall_s"] = round(time.perf_counter() - t_proc, 1)
        shutil.rmtree(workdir, ignore_errors=True)
    return rec


def write_receipt(out: str, rec: dict[str, Any]) -> str:
    os.makedirs(os.path.join(out, RUNS), exist_ok=True)
    rec = {"manifest": rec.pop("manifest", {}), **rec}
    rec["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = L.scrub(json.dumps(rec, indent=1), L.hf_secrets())
    path = os.path.join(out, RUNS, f"{rec['tag']}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    os.replace(tmp, path)
    return path


def cmd_point(a: argparse.Namespace) -> int:
    if a.synthetic:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    rec = measure_point(a)
    path = write_receipt(a.out, rec)
    print(f"RECEIPT {path} status={rec.get('status')} peak_gib_for_fit={rec.get('peak_gib_for_fit')} "
          f"s_per_step={rec.get('s_per_step')}", flush=True)
    return 0 if rec.get("status") in ("ok", "oom") else 1


# ------------------------------------------------------------ orchestrator
def _sh(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return 127, f"{type(exc).__name__}: {exc}"


def gpu_used_mib() -> int | None:
    rc, out = _sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if rc != 0:
        return None
    try:
        return int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


NETWORK_FS = re.compile(r"nfs|fuse|moose|cifs|smb|9p|ceph|lustre|gluster|beegfs", re.I)


def fs_type(path: str) -> str | None:
    """Filesystem type of the mount holding ``path`` (Linux; None elsewhere)."""
    try:
        best, kind = "", None
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and os.path.abspath(path).startswith(parts[1]) and len(parts[1]) > len(best):
                    best, kind = parts[1], parts[2]
        return kind
    except OSError:
        return None


def preflight(a: argparse.Namespace, points: list[L.Point], presets: list[L.Preset]) -> dict[str, Any]:
    """Andon checks before anything is spent. Returns facts; raises SystemExit(2) on a halt."""
    facts: dict[str, Any] = {"synthetic": bool(a.synthetic)}
    if a.synthetic:
        return facts
    problems: list[str] = []
    rc, out = _sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                   "--format=csv,noheader,nounits"])
    if rc != 0:
        problems.append(f"nvidia-smi failed: {out.strip()[:200]}")
    else:
        facts["gpu"] = out.strip()
        used = gpu_used_mib()
        if used is not None and used > 1500:
            problems.append(f"{used} MiB already in use on the GPU: another process would pollute every NVML peak")
    rc, out = _sh([sys.executable, "-c",
                   "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"], timeout=120)
    if rc != 0:
        problems.append("torch.cuda.is_available() is False (the pod's GPU is broken or the image is wrong)")
    rc, out = _sh([sys.executable, "-c", "import pynvml"])
    if rc != 0:
        problems.append("pynvml (nvidia-ml-py) is not installed: no NVML peaks, the whole point of this sweep")
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    os.makedirs(hf_home, exist_ok=True)
    kind = fs_type(hf_home)
    facts["hf_home"], facts["hf_home_fs"] = hf_home, kind
    if kind and NETWORK_FS.search(kind):
        problems.append(f"HF_HOME={hf_home} is on a network filesystem ({kind}); it must live on the container disk")
    free_gb = shutil.disk_usage(hf_home).free / 1e9
    need = 0.0
    names = [p.name for p in presets]
    sizes = [L.repo_params_b(n, next(p.params_b_name for p in presets if p.name == n)) * 2.0 for n in names]
    if sizes:
        need = max(sizes) + 15.0
    facts["hf_free_gb"], facts["hf_need_gb_min"] = round(free_gb, 1), round(need, 1)
    if free_gb < need:
        problems.append(f"only {free_gb:.0f} GB free under HF_HOME; the largest preset needs ~{need:.0f} GB "
                        "(create the pod with RUNPOD_CONTAINER_DISK_GB=200)")
    if any(p.unsloth for p in points):
        import importlib.metadata as md

        try:
            facts["unsloth"] = md.version("unsloth")
        except md.PackageNotFoundError:
            if a.allow_missing_unsloth:
                facts["unsloth"] = None
            else:
                problems.append("unsloth is not installed but Unsloth-on points are planned "
                                "(pip install -e .[unsloth], or --allow-missing-unsloth to drop them)")
    facts["hf_token_present"] = bool(L.hf_secrets())
    if problems:
        print("ANDON: " + "; ".join(problems), flush=True)
        L.write_json(os.path.join(a.out, "halt.json"), {"halted": True, "problems": problems, "facts": facts})
        raise SystemExit(2)
    return facts


def hf_access(model_id: str) -> str | None:
    """None when the repo is readable with the current environment, else why not."""
    try:
        from huggingface_hub import auth_check
    except ImportError:
        return None  # an old huggingface_hub cannot tell us: try the download and let it fail loudly
    try:
        auth_check(model_id)
        return None
    except Exception as exc:  # noqa: BLE001 - GatedRepoError, RepositoryNotFoundError, offline
        return L.scrub(f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}", L.hf_secrets())


def fetch_repo(model_id: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model",
                                                "tokenizer*"])


def purge_repo(model_id: str) -> str:
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    revs = [r.commit_hash for repo in info.repos if repo.repo_id == model_id for r in repo.revisions]
    if not revs:
        return "nothing cached"
    strat = info.delete_revisions(*revs)
    freed = strat.expected_freed_size_str
    strat.execute()
    return f"freed {freed}"


class Prefetcher:
    """Download the next preset in a background thread while the current one measures."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.thread: threading.Thread | None = None
        self.model: str | None = None
        self.error: str | None = None

    def start(self, model_id: str, repo_gb: float, hf_home: str) -> None:
        if not self.enabled or self.thread is not None:
            return
        free_gb = shutil.disk_usage(hf_home).free / 1e9
        if free_gb < repo_gb * 1.1 + 10.0:
            print(f"prefetch of {model_id} skipped: {free_gb:.0f} GB free, need ~{repo_gb * 1.1 + 10:.0f}", flush=True)
            return
        self.model, self.error = model_id, None

        def work() -> None:
            try:
                fetch_repo(model_id)
            except Exception as exc:  # noqa: BLE001
                self.error = L.scrub(f"{type(exc).__name__}: {str(exc)[:300]}", L.hf_secrets())

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def finish(self, model_id: str) -> str | None:
        """Block until ``model_id`` is on disk (waiting for its prefetch, or fetching now)."""
        if self.thread is not None and self.model == model_id:
            self.thread.join()
            self.thread = None
            if self.error is None:
                return None
        try:
            fetch_repo(model_id)
            return None
        except Exception as exc:  # noqa: BLE001
            return L.scrub(f"{type(exc).__name__}: {str(exc)[:300]}", L.hf_secrets())


def run_child(a: argparse.Namespace, pt: L.Point, est_s: float) -> dict[str, Any]:
    """Spawn one ``point`` process; make sure a receipt exists afterwards."""
    cmd = [sys.executable, os.path.abspath(__file__), "point", "--out", a.out, "--preset", pt.preset,
           "--batch", str(pt.batch), "--unsloth", "on" if pt.unsloth else "off", "--steps", str(a.steps),
           "--seed", str(a.seed)]
    if a.synthetic:
        cmd.append("--synthetic")
    log_dir = os.path.join(a.out, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{pt.tag}.log")
    timeout = max(900.0, 4.0 * est_s) if not a.synthetic else 900.0
    t0 = time.perf_counter()
    status = None
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc, status = -9, "timeout"
    wall = time.perf_counter() - t0
    # scrub the log in place (the Hub client can echo URLs)
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        with open(log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(L.scrub(text, L.hf_secrets()))
    except OSError:
        text = ""
    receipt = os.path.join(a.out, RUNS, f"{pt.tag}.json")
    if not os.path.exists(receipt):
        write_receipt(a.out, {
            "mode": "e3_point", "tag": pt.tag, "preset": pt.preset, "model": pt.model_id, "batch": pt.batch,
            "unsloth_requested": pt.unsloth, "steps": a.steps, "dry_run": bool(a.synthetic),
            "status": status or "crashed", "returncode": rc, "wall_s": round(wall, 1),
            "log_tail": L.scrub(text[-3000:], L.hf_secrets()),
            "manifest": {"git_sha": git_sha(), "dry_run": bool(a.synthetic)}})
    return L.read_json(receipt)


def wait_gpu_free(limit_s: float = 60.0) -> bool:
    """After a child exits its GPU memory must be released; a zombie pollutes the next NVML peak."""
    t_end = time.time() + limit_s
    while time.time() < t_end:
        used = gpu_used_mib()
        if used is None or used < 1500:
            return True
        time.sleep(3)
    return False


#: A point with one of these receipts is done; anything else (dropped, skipped,
#: error, crashed, timeout) is retried when the run is resumed.
FINAL_STATUSES = ("ok", "oom")


def receipt_is_final(out: str, tag: str) -> bool:
    path = os.path.join(out, RUNS, f"{tag}.json")
    if not os.path.exists(path):
        return False
    try:
        return L.read_json(path).get("status") in FINAL_STATUSES
    except (OSError, ValueError):
        return False


def skipped_receipt(a: argparse.Namespace, pt: L.Point, status: str, why: str, extra: dict | None = None) -> None:
    write_receipt(a.out, {"mode": "e3_point", "tag": pt.tag, "preset": pt.preset, "model": pt.model_id,
                          "batch": pt.batch, "unsloth_requested": pt.unsloth, "steps": a.steps,
                          "dry_run": bool(a.synthetic), "status": status, "reason": why,
                          "manifest": {"git_sha": git_sha(), "dry_run": bool(a.synthetic)}, **(extra or {})})


def cmd_run(a: argparse.Namespace) -> int:
    if a.synthetic:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.makedirs(os.path.join(a.out, RUNS), exist_ok=True)
    presets, points = select_grid(a)
    t_start = time.time()
    facts = preflight(a, points, presets)
    if facts.get("unsloth", True) is None:
        points = [p for p in points if not p.unsloth]
    by_preset = {p.name: p for p in presets}

    # gated / unreachable repos are skipped up front, with a receipt per point
    if not a.synthetic:
        blocked: dict[str, str] = {}
        for p in presets:
            why = hf_access(p.model_id)
            if why:
                blocked[p.name] = why
        for pt in list(points):
            if pt.preset in blocked and not receipt_is_final(a.out, pt.tag):
                skipped_receipt(a, pt, "skipped_no_access", blocked[pt.preset])
        points = [p for p in points if p.preset not in blocked]
        facts["blocked_presets"] = blocked

    plan = L.plan_budget(presets, points, a.budget_usd, a.usd_per_hour, a.steps, a.dl_mbps, not a.no_prefetch)
    keep = points if a.no_guard else plan.keep
    dropped_tags: set[str] = set()
    L.write_json(os.path.join(a.out, "plan.json"), {**plan_report(a), "preflight": facts,
                                                    "no_guard": bool(a.no_guard)})
    print(f"plan: {len(keep)} of {len(points)} points kept (~{plan.est_seconds / 3600:.2f} h, "
          f"~${plan.est_usd:.2f}); dropped {len(plan.dropped)}", flush=True)
    if not a.no_guard:
        for pt in plan.dropped:
            dropped_tags.add(pt.tag)
            if not receipt_is_final(a.out, pt.tag):
                skipped_receipt(a, pt, "dropped_by_budget_guard", f"planned drop (tier {pt.tier})")

    guard_log: list[dict[str, Any]] = []
    remaining = list(keep)
    done_est = done_actual = 0.0
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pre = Prefetcher(enabled=not a.no_prefetch and not a.synthetic)
    order = [p for p in presets if any(pt.preset == p.name for pt in keep)]
    budget_s = plan.budget_seconds
    halted = None

    for i, preset in enumerate(order):
        my_points = [pt for pt in remaining if pt.preset == preset.name]
        if not my_points:
            continue
        # make sure this preset is on disk; start the next one's download
        if not a.synthetic:
            err = pre.finish(preset.model_id)
            if err:
                for pt in my_points:
                    skipped_receipt(a, pt, "skipped_download_failed", err)
                remaining = [r for r in remaining if r.preset != preset.name]
                continue
            if i + 1 < len(order):
                nxt = order[i + 1]
                pre.start(nxt.model_id, L.repo_params_b(nxt.name, nxt.params_b_name) * 2.0, hf_home)
        oom_at: dict[bool, int] = {}
        for pt in my_points:
            remaining = [r for r in remaining if r != pt]
            if pt.tag in dropped_tags:
                continue
            if receipt_is_final(a.out, pt.tag):
                print(f"skip {pt.tag} (measured receipt exists)", flush=True)
                continue
            if a.stop_after_epoch and time.time() >= a.stop_after_epoch:
                skipped_receipt(a, pt, "dropped_hard_stop", "past --stop-after-epoch")
                continue
            # runtime guard: actual timings can overrun the planning figures
            est = L.estimate_point_seconds(preset, pt.batch, pt.unsloth, a.steps)
            if not a.no_guard:
                scale = max(0.5, done_actual / done_est) if done_est > 0 else 1.0
                elapsed = time.time() - t_start
                dropped_now = L.runtime_guard([pt] + remaining, by_preset, a.steps, elapsed, budget_s, scale)
                if dropped_now:
                    why = (f"run-time guard: elapsed {elapsed:.0f}s + remaining estimate (x{scale:.2f} "
                           f"observed/estimated) exceeds {budget_s:.0f}s")
                    for v in dropped_now:
                        remaining = [r for r in remaining if r != v]
                        dropped_tags.add(v.tag)
                        if not receipt_is_final(a.out, v.tag):
                            skipped_receipt(a, v, "dropped_by_budget_guard", why)
                    guard_log.append({"at_elapsed_s": round(elapsed), "scale": round(scale, 2),
                                      "dropped": [v.tag for v in dropped_now]})
                    print(f"GUARD dropped {len(dropped_now)} point(s): {[v.tag for v in dropped_now][:4]}...",
                          flush=True)
                    if pt in dropped_now:
                        continue
            # monotone OOM: a larger batch cannot fit if a smaller one did not (never for tier 0)
            if (not a.no_monotone_skip and pt.tier != 0 and pt.unsloth in oom_at
                    and pt.batch > oom_at[pt.unsloth]):
                skipped_receipt(a, pt, "skipped_monotone_oom",
                                f"batch {oom_at[pt.unsloth]} already OOMed with unsloth={pt.unsloth}")
                continue
            print(f"[{time.strftime('%H:%M:%S')}] {pt.tag} (tier {pt.tier}, est {est:.0f}s)", flush=True)
            rec = run_child(a, pt, est)
            done_est += est
            done_actual += rec.get("wall_s") or est
            print(f"  -> {rec.get('status')} peak={rec.get('peak_gib_for_fit')} s/step={rec.get('s_per_step')} "
                  f"wall={rec.get('wall_s')}s", flush=True)
            if rec.get("status") == "oom":
                oom_at[pt.unsloth] = min(pt.batch, oom_at.get(pt.unsloth, pt.batch))
            if not a.synthetic and not wait_gpu_free():
                halted = f"GPU memory not released after {pt.tag}; a stray process would pollute every later peak"
                print("ANDON: " + halted, flush=True)
                break
        if halted:
            break
        if not a.synthetic and not a.no_purge:
            print(f"purge {preset.model_id}: {purge_repo(preset.model_id)}", flush=True)

    summary = build_summary(a.out)
    summary["run"] = {"elapsed_s": round(time.time() - t_start), "est_s_planned": round(plan.est_seconds),
                      "guard_log": guard_log, "halted": halted, "preflight": facts,
                      "budget_usd": a.budget_usd, "usd_per_hour": a.usd_per_hour,
                      "est_usd_actual": round((time.time() - t_start) / 3600 * a.usd_per_hour, 2)}
    L.write_json(os.path.join(a.out, "summary.json"), summary)
    print_summary(summary)
    return 2 if halted else 0


# ------------------------------------------------------------------ summary
def load_receipts(out: str) -> list[dict[str, Any]]:
    d = os.path.join(out, RUNS)
    recs = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if name.startswith("e3_") and name.endswith(".json"):
                recs.append(L.read_json(os.path.join(d, name)))
    return recs


def build_summary(out: str) -> dict[str, Any]:
    recs = load_receipts(out)
    rows = []
    for r in recs:
        pd, pa = r.get("predicted_default") or {}, r.get("predicted_arch") or {}
        peak = r.get("peak_gib_for_fit")
        rows.append({
            "tag": r["tag"], "preset": r.get("preset"), "batch": r.get("batch"),
            "unsloth": r.get("unsloth_requested"), "unsloth_active": r.get("unsloth_active"),
            "status": r.get("status"), "window": r.get("window"),
            "peak_gib": peak, "peak_source": r.get("peak_source"),
            "nvml_peak_gib": r.get("nvml_peak_gib"), "torch_reserved_gib": r.get("torch_max_reserved_gib"),
            "torch_allocated_gib": r.get("torch_max_allocated_gib"),
            "paged_state_gib": r.get("optimizer_paged_state_gib"), "s_per_step": r.get("s_per_step"),
            "predicted_default_gb": pd.get("total_gb"), "predicted_arch_gb": pa.get("total_gb"),
            "err_default_pct": (None if not (peak and pd.get("total_gb")) else round(100 * (pd["total_gb"] - peak) / peak, 1)),
            "err_arch_pct": (None if not (peak and pa.get("total_gb")) else round(100 * (pa["total_gb"] - peak) / peak, 1)),
            "auto_batch_choice": r.get("auto_batch_choice"),
            "oom_retries": r.get("oom_retries"), "plateau_ok": r.get("plateau_ok"), "shape_ok": r.get("shape_ok"),
            "reason": r.get("reason"),
        })
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    manifest = next((r.get("manifest") for r in recs if r.get("status") == "ok"), None)
    return {"mode": "e3_summary", "points": len(rows), "status_counts": counts, "manifest": manifest, "rows": rows,
            "dry_run": any(r.get("dry_run") for r in recs)}


def print_summary(s: dict[str, Any]) -> None:
    print(f"\nE3 summary: {s['points']} receipts {s['status_counts']}")
    print(f"{'tag':44s} {'status':10s} {'peak':>7s} {'pred.def':>8s} {'pred.arch':>9s} {'s/step':>7s}")
    for r in s["rows"]:
        print(f"{r['tag']:44s} {str(r['status']):10s} {str(r['peak_gib']):>7s} "
              f"{str(None if r['predicted_default_gb'] is None else round(r['predicted_default_gb'], 1)):>8s} "
              f"{str(None if r['predicted_arch_gb'] is None else round(r['predicted_arch_gb'], 1)):>9s} "
              f"{str(r['s_per_step']):>7s}")


def cmd_summarize(a: argparse.Namespace) -> int:
    s = build_summary(a.out)
    prev = os.path.join(a.out, "summary.json")
    if os.path.exists(prev):
        try:
            s["run"] = L.read_json(prev).get("run")
        except (OSError, ValueError):
            s["run"] = None
    L.write_json(prev, s)
    print_summary(s)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "arch":
        return cmd_arch(a)
    if a.cmd == "point":
        return cmd_point(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "summarize":
        return cmd_summarize(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
