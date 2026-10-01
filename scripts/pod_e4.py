"""E4 orchestrator: where does full fine-tuning beat QLoRA? (runs on the pod, or CPU with --dry-run)

One process per command; every run is one ``pod_block_engine.py train --dataset code`` subprocess
that writes one receipt, so a stage resumes by skipping tags whose receipt exists.

    prep        OpenCodeInstruct -> train / eval split (+ tokenizer length stats)
    precheck    untrained base model on the eval set; abort gate 0.10 < pass@1 < 0.80   (exit 4 = abort)
    stage       the stage's runs in priority order under the time-based budget guard
    gate        statistics + the pre-registered verdict                                  (exit 3 = stop)
    run         prep -> precheck -> stage -> gate for one size
    dry-run     the whole pipeline on CPU with a tiny model for 3 steps (no code is executed)

The 7B stage refuses to start unless ``stage_e4_3b.json`` holds a PASS premise verdict (or
``E4_PREMISE_OVERRIDE=<reason>`` is set; the reason is written into the budget receipt).

Pre-registration: docs/receipts/2026-10-e4-code/README.md. Budget guard: ``e4_lib.BudgetGuard``;
drop order is the reverse of ``e4_lib.plan_for`` (also in the README).

Env: E4_STEPS (5000), E4_BATCH (4), E4_SEQ (512), E4_SEEDS_3B ("0 1 2"), E4_SEEDS_7B ("0 1"),
E4_MODEL_3B / E4_MODEL_7B, E4_BUDGET_USD_3B (4.80) / _7B (4.50), E4_RATE_USD_H (0.90),
E4_START_EPOCH (when the pod's billing began; default: first command of the stage),
E4_RESERVE_S (600), E4_N_EVAL (500), E4_MAX_NEW_TOKENS (512).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 - runs this repo's own driver scripts, argv lists only
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_lib as E4  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "pod_block_engine.py")
SUMMARY = os.path.join(HERE, "pod_e4_summary.py")

MODELS = {"3b": "Qwen/Qwen2.5-3B", "7b": "Qwen/Qwen2.5-7B"}
CAPS_USD = {"3b": 4.80, "7b": 4.50}  # 3B raised from 4.00 by the Director, 2026-10-01
ARM_ARGS = {  # the stage-d arms: each arm at the library's own learning rate
    "default": ["--engine", "default", "--lr-default"],
    "qlora": ["--qlora", "--lr-default"],
    "block_k5": ["--engine", "block", "--k", "5", "--lr-default"],
    "galore": ["--engine", "default", "--lr-default", "--ceiling", "8",
               "--galore", "galore_adamw_8bit_layerwise"],
}
EXIT_PRECHECK_ABORT = 4
EXIT_GATE_STOP = 3


@dataclass
class Cfg:
    out: str
    size: str
    model: str
    steps: int
    batch: int
    seq: int
    seeds: tuple[int, ...]
    cap_usd: float
    rate_usd_h: float
    reserve_s: float
    n_eval: int
    max_new_tokens: int
    eval_batch: int = 50
    dry_run: bool = False
    synthetic_train: int = 32
    start_epoch: float = 0.0
    load_s: float = 60.0

    @property
    def runs(self) -> str:
        return os.path.join(self.out, "runs")

    def receipt(self, tag: str) -> str:
        return os.path.join(self.runs, f"{tag}.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, rec: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)


def run_py(script: str, args: list[str], env_extra: dict | None = None) -> int:
    env = dict(os.environ)
    env.update(env_extra or {})
    # nosec B603: argv is [this interpreter, a script in this repo, flags built here]; no shell.
    return subprocess.run([sys.executable, script, *args], env=env, check=False).returncode  # nosec B603


def driver(cfg: Cfg, mode: str, extra: list[str], env_extra: dict | None = None) -> int:
    base = [mode, "--out", cfg.out, "--dataset", "code", "--max-new-tokens", str(cfg.max_new_tokens),
            "--eval-batch", str(cfg.eval_batch)]
    if not cfg.dry_run:
        base.append("--allow-exec")  # model-generated code runs only on the pod
    return run_py(DRIVER, base + extra, env_extra)


def make_cfg(size: str, ns: argparse.Namespace, out: str | None = None) -> Cfg:
    env = os.environ
    dry = bool(getattr(ns, "dry_run", False))
    seeds_env = env.get(f"E4_SEEDS_{size.upper()}", "0 1 2" if size == "3b" else "0 1")
    seeds = tuple(int(x) for x in seeds_env.split())
    cfg = Cfg(
        out=out or ns.out, size=size, model=env.get(f"E4_MODEL_{size.upper()}", MODELS[size]),
        steps=int(env.get("E4_STEPS", 5000)), batch=int(env.get("E4_BATCH", 4)), seq=int(env.get("E4_SEQ", 512)),
        seeds=seeds, cap_usd=float(env.get(f"E4_BUDGET_USD_{size.upper()}", CAPS_USD[size])),
        rate_usd_h=float(env.get("E4_RATE_USD_H", 0.90)), reserve_s=float(env.get("E4_RESERVE_S", 600)),
        n_eval=int(env.get("E4_N_EVAL", E4.DEFAULT_N_EVAL)),
        max_new_tokens=int(env.get("E4_MAX_NEW_TOKENS", 512)), dry_run=dry,
        load_s=60.0 if size == "3b" else 120.0)
    if dry:
        cfg.model = getattr(ns, "model", None) or "HuggingFaceTB/SmolLM2-135M"
        cfg.steps, cfg.batch, cfg.seq = 3, 2, 256
        cfg.seeds, cfg.n_eval, cfg.max_new_tokens, cfg.eval_batch = (0, 1), 8, 48, 8
    os.makedirs(cfg.runs, exist_ok=True)
    cfg.start_epoch = _start_epoch(cfg)
    return cfg


def _start_epoch(cfg: Cfg) -> float:
    """Billing clock: E4_START_EPOCH, else a marker written by the first command of the stage."""
    if os.environ.get("E4_START_EPOCH"):
        return float(os.environ["E4_START_EPOCH"])
    marker = os.path.join(cfg.out, f".e4_start_{cfg.size}")
    if os.path.exists(marker):
        with open(marker, encoding="utf-8") as fh:
            return float(fh.read().strip())
    now = time.time()
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(str(now))
    return now


# ------------------------------------------------------------------ commands
def fetch(model: str, *, weights: bool = True) -> int:
    """Download a model into the HF cache. ``weights=False`` fetches only the tokenizer / config
    files (the token-length stats need nothing more)."""
    if os.path.isdir(model):
        return 0
    patterns = ["*.json", "*.txt", "*.jinja", "*.model", "tokenizer*"] + (["*.safetensors"] if weights else [])
    code = f"from huggingface_hub import snapshot_download as s; s({model!r}, allow_patterns={patterns!r})"
    rc = subprocess.run([sys.executable, "-c", code], check=False).returncode  # nosec B603
    if rc != 0:
        log(f"download of {model} failed (re-run to resume)")
    return rc


def purge_other_models(keep: str) -> None:
    """Drop the other stage's model from the HF cache (container-disk headroom)."""
    drop = [m for m in MODELS.values() if m != keep]
    code = (
        "import sys\nfrom huggingface_hub import scan_cache_dir\ninfo = scan_cache_dir()\n"
        "revs = [r.commit_hash for repo in info.repos if repo.repo_id in sys.argv[1:] for r in repo.revisions]\n"
        "if revs:\n    s = info.delete_revisions(*revs)\n    print('purge: freeing', s.expected_freed_size_str)\n    s.execute()\n")
    subprocess.run([sys.executable, "-c", code, *drop], check=False)  # nosec B603


def cmd_prep(cfg: Cfg) -> int:
    if not os.path.exists(os.path.join(cfg.out, "prep_code.json")):
        if fetch(cfg.model, weights=False) != 0:  # tokenizer only: the fit filter tokenizes every row
            return 2
        extra = ["--n-eval", str(cfg.n_eval), "--model", cfg.model, "--seq", str(cfg.seq),
                 "--min-train", str(cfg.steps * cfg.batch)]
        if cfg.dry_run:
            extra += ["--synthetic", "--n-train", str(cfg.synthetic_train)]
        rc = driver(cfg, "prep", extra)
        if rc != 0:
            log("prep failed")
            return 2
    lengths = os.path.join(cfg.runs, f"lengths_code_{E4.slug(cfg.model)}.json")
    if not os.path.exists(lengths):
        if fetch(cfg.model, weights=False) != 0:  # tokenizer only
            return 2
        rc = driver(cfg, "lengths", ["--model", cfg.model, "--seq", str(cfg.seq), "--batch", str(cfg.batch),
                                     "--steps", str(cfg.steps)])
        if rc != 0:  # andon: some text would still be cut at --seq
            log("lengths check failed: the fit filter left texts over --seq")
            return 2
    rec = read_json(lengths)
    if rec:
        log(f"truncated at {cfg.seq}: {rec.get('truncated_fraction_at_seq')}; real tokens over the run "
            f"{rec.get('real_tokens_over_run')}, epochs {rec.get('epochs')}")
    return 0


def cmd_precheck(cfg: Cfg) -> int:
    base = cfg.receipt(f"base_code_{E4.slug(cfg.model)}")
    if not os.path.exists(base):
        if cfg.size == "7b" and not cfg.dry_run:
            purge_other_models(cfg.model)  # free the 3B copy before the 7B download
        if fetch(cfg.model) != 0:
            return 2
        driver(cfg, "base", ["--model", cfg.model])  # exit status is read from the receipt below
    rec = read_json(base)
    if rec is None:
        log("precheck: the base run left no receipt")
        return 2
    pre = rec.get("precheck") or {}
    log(f"PRECHECK {cfg.size}: {pre.get('verdict')} pass@1={rec.get('pass_at_1')} "
        f"heldout_loss={rec.get('heldout_loss')} {pre.get('reasons')}")
    write_json(os.path.join(cfg.out, f"precheck_{cfg.size}.json"),
               {"size": cfg.size, "model": cfg.model, **pre, "pass_at_1": rec.get("pass_at_1"),
                "heldout_loss": rec.get("heldout_loss")})
    return 0 if pre.get("ok") else EXIT_PRECHECK_ABORT


def train_args(cfg: Cfg, spec: E4.RunSpec) -> list[str]:
    return (["--tag", spec.tag, "--model", cfg.model, "--seed", str(spec.seed), "--steps", str(cfg.steps),
             "--batch", str(cfg.batch), "--seq", str(cfg.seq)] + ARM_ARGS[spec.arm])


def _require_gates(cfg: Cfg) -> int | None:
    """Pre-registered gates the stage depends on; None = clear to run."""
    if cfg.dry_run:
        return None
    pre = read_json(os.path.join(cfg.out, f"precheck_{cfg.size}.json"))
    if not pre or not pre.get("ok"):
        log(f"refusing to train: no passing precheck_{cfg.size}.json (run `precheck {cfg.size}` first)")
        return EXIT_PRECHECK_ABORT
    if cfg.size == "7b":
        stage3 = read_json(os.path.join(cfg.out, "stage_e4_3b.json"))
        passed = bool(stage3 and stage3.get("verdict", {}).get("verdict") == "PASS")
        if not passed and not os.environ.get("E4_PREMISE_OVERRIDE"):
            log("refusing to run the 7B stage: the 3B premise gate has no PASS verdict in stage_e4_3b.json. "
                "Per the pre-registration the 7B stage does not run (E4_PREMISE_OVERRIDE=<reason> to override).")
            return EXIT_GATE_STOP
    return None


def cmd_stage(cfg: Cfg) -> int:
    blocked = _require_gates(cfg)
    if blocked is not None:
        return blocked
    plan = E4.plan_for(cfg.size, cfg.seeds)
    if cfg.dry_run and cfg.size == "7b" and not os.environ.get("E4_DRY_RUN_GALORE"):
        plan = [s for s in plan if s.arm != "galore"]  # GaLore + 8-bit paging need CUDA
    guard = E4.BudgetGuard(cap_usd=cfg.cap_usd, rate_usd_h=cfg.rate_usd_h, start_epoch=cfg.start_epoch,
                           reserve_s=cfg.reserve_s)
    base = read_json(cfg.receipt(f"base_code_{E4.slug(cfg.model)}")) or {}
    base_eval_s = float(base.get("eval_s") or 300.0)
    measured: dict[str, float] = {}
    for spec in plan:
        r = read_json(cfg.receipt(spec.tag))
        if r and r.get("status") == "ok" and r.get("wall_s"):
            measured[spec.arm] = float(r["wall_s"])
    decisions = []
    override = os.environ.get("E4_PREMISE_OVERRIDE")
    log(f"stage {cfg.size}: {len(plan)} runs, {cfg.steps} steps x batch {cfg.batch} x {cfg.seq} tokens; budget "
        f"${cfg.cap_usd:.2f} at ${cfg.rate_usd_h:.2f}/h, deadline in "
        f"{(guard.deadline_epoch - time.time()) / 60:.0f} min")
    for spec in plan:
        existing = read_json(cfg.receipt(spec.tag))
        if existing is not None:
            log(f"skip {spec.tag} (receipt exists, status {existing.get('status')})")
            continue
        est = E4.estimate_run_s(
            spec, cfg.steps, eval_s=base_eval_s * (2.0 if spec.arm == "qlora" else 1.0), load_s=cfg.load_s,
            measured_wall_s=measured.get(spec.arm) if not cfg.dry_run else None)
        ok, why = guard.admit(spec, est)
        decisions.append({"tag": spec.tag, "est_s": round(est), "admitted": ok, "why": why})
        if not ok:
            log(f"DROP {spec.tag}: {why}")
            continue
        log(f"run {spec.tag} (est {est / 60:.0f} min)")
        t0 = time.time()
        rc = driver(cfg, "train", train_args(cfg, spec), {"ARM": spec.arm})
        r = read_json(cfg.receipt(spec.tag))
        if r is None:
            log(f"run {spec.tag} exited {rc} without a receipt")
            continue
        if r.get("status") == "ok" and r.get("wall_s"):
            measured[spec.arm] = float(r["wall_s"])
        log(f"done {spec.tag}: status={r.get('status')} pass@1={r.get('pass_at_1')} "
            f"heldout_after={r.get('heldout_after')} wall={time.time() - t0:.0f}s")
        write_budget(cfg, guard, decisions, override)
        run_py(SUMMARY, ["--out", cfg.out, "--size", cfg.size, "--boot", "2000", "--model", cfg.model])  # provisional
    write_budget(cfg, guard, decisions, override)
    return 0


def write_budget(cfg: Cfg, guard: E4.BudgetGuard, decisions: list[dict], override: str | None) -> None:
    write_json(os.path.join(cfg.out, f"budget_{cfg.size}.json"), {
        "cap_usd": cfg.cap_usd, "rate_usd_h": cfg.rate_usd_h, "start_epoch": cfg.start_epoch,
        "deadline_epoch": guard.deadline_epoch, "reserve_s": cfg.reserve_s,
        "elapsed_h": round((time.time() - cfg.start_epoch) / 3600, 3),
        "spent_usd_estimate": round((time.time() - cfg.start_epoch) / 3600 * cfg.rate_usd_h, 2),
        "drop_order": [s.tag for s in reversed(E4.plan_for(cfg.size, cfg.seeds))],
        "decisions": decisions, "dropped": guard.dropped, "premise_override": override})


def cmd_gate(cfg: Cfg) -> int:
    rc = run_py(SUMMARY, ["--out", cfg.out, "--size", cfg.size, "--model", cfg.model])
    rec = read_json(os.path.join(cfg.out, f"stage_e4_{cfg.size}.json")) or {}
    v = (rec.get("verdict") or {}).get("verdict")
    log(f"GATE {cfg.size}: {v}")
    return 0 if v == "PASS" else (EXIT_GATE_STOP if rc in (0, EXIT_GATE_STOP) else rc)


def cmd_run(cfg: Cfg) -> int:
    for step in (cmd_prep, cmd_precheck, cmd_stage, cmd_gate):
        rc = step(cfg)
        if rc != 0:
            log(f"stopping after {step.__name__}: exit {rc}")
            return rc
    return 0


def cmd_dry_run(ns: argparse.Namespace) -> int:
    """The whole pipeline on CPU, tiny model, 3 steps; no generated code is executed."""
    results = {}
    for size in ("3b", "7b"):
        cfg = make_cfg(size, ns, out=os.path.join(ns.out, f"dry_{size}"))
        # Share the prep and tokenizer stats between the two stages.
        for step in (cmd_prep, cmd_precheck, cmd_stage):
            rc = step(cfg)
            if rc != 0:
                log(f"dry-run {size}: {step.__name__} exit {rc}")
                return rc
        run_py(SUMMARY, ["--out", cfg.out, "--size", size, "--boot", "500", "--model", cfg.model])
        rec = read_json(os.path.join(cfg.out, f"stage_e4_{size}.json")) or {}
        results[size] = {"verdict": (rec.get("verdict") or {}).get("verdict"), "arms": sorted(rec.get("arms", {}))}
    log("DRY-RUN OK " + json.dumps(results))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prep", "precheck", "stage", "gate", "run", "dry-run"])
    ap.add_argument("size", nargs="?", choices=["3b", "7b"], help="not used by dry-run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="dry-run only: a tiny model id or local dir")
    ns = ap.parse_args(argv)
    if ns.command == "dry-run":
        ns.dry_run = True
        return cmd_dry_run(ns)
    if ns.size is None:
        ap.error(f"{ns.command} needs a size (3b | 7b)")
    ns.dry_run = False
    cfg = make_cfg(ns.size, ns)
    return {"prep": cmd_prep, "precheck": cmd_precheck, "stage": cmd_stage, "gate": cmd_gate,
            "run": cmd_run}[ns.command](cfg)


if __name__ == "__main__":
    sys.exit(main())
