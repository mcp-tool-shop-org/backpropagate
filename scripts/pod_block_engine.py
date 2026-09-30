"""Evidence driver for Engine B (block-coordinate AdamW) — runs on a GPU pod.

Driven by ``scripts/pod_block_engine.sh``; every mode is one process and writes
one JSON receipt, so each run is independently resumable (the shell skips a
tag whose receipt exists).

Modes
-----
    prep       dolly-15k -> <out>/train.jsonl (400) + <out>/heldout.jsonl (150, disjoint)
    base       held-out loss of the untrained model; checks floor < loss < ceiling
    train      one library Trainer run (mode='full'; engine 'block' or 'default',
               or mode='lora' QLoRA with --qlora), held-out loss after, s/step,
               peak VRAM (optionally under a VRAM cap), per-block-visit peaks,
               the fit projection for the same config, optional save->reload->generate
    summarize  aggregate the receipts of one stage into a PASS/FAIL receipt

Data: databricks/databricks-dolly-15k (CC BY-SA 3.0), context folded into the
user turn, fixed-seed shuffle — the same split as the offload quality check on
feat/offload-7b (scripts/offload_quality.py). Held-out loss is token-weighted
full-sequence cross-entropy on the exact text the library trains on
(DatasetLoader.to_hf_dataset), truncated to --seq.

Nothing here is a unit test: it measures. Seeds, versions, GPU and the git SHA
(from the shell) go into every receipt.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["prep", "base", "train", "summarize"])
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default=None)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--engine", choices=["block", "default"], default="block")
ap.add_argument("--qlora", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--order", default="random")
ap.add_argument("--writeback", default="stochastic")
ap.add_argument("--freeze-embeddings", action="store_true")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--vram-cap-gb", type=float, default=0.0, help="GiB; 0 = uncapped")
ap.add_argument("--save-reload", action="store_true")
ap.add_argument("--no-eval", action="store_true")
ap.add_argument("--gen-inprocess", action="store_true",
                help="generate from the trained in-memory model (no save/reload)")
ap.add_argument("--stage", default=None, help="summarize: a | b | c | a2")
ap.add_argument("--git-sha", default=os.environ.get("GIT_SHA", "unknown"))
ap.add_argument("--n-train", type=int, default=400)
ap.add_argument("--n-heldout", type=int, default=150)
ap.add_argument("--synthetic", action="store_true",
                help="prep: write a tiny synthetic split instead of downloading dolly (driver dry run)")
args = ap.parse_args()

OUT = args.out
RUNS = os.path.join(OUT, "runs")
os.makedirs(RUNS, exist_ok=True)
TRAIN = os.path.join(OUT, "train.jsonl")
HELD = os.path.join(OUT, "heldout.jsonl")
GIB = 2**30
# Trainer output (its end-of-run checkpoint) and the save->reload copy can live
# on fast local disk while receipts stay on the (network) volume.
WORKDIR = os.environ.get("BP_WORK_DIR") or os.path.join(OUT, "work")
SAVEDIR = os.environ.get("BP_SAVE_DIR") or WORKDIR


def slug(model: str) -> str:
    """File-name form of a model id ("Qwen/Qwen2.5-7B" -> "Qwen_Qwen2.5-7B")."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in model)


def dump(path: str, rec: dict) -> None:
    rec["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, path)
    print("RECEIPT " + json.dumps(rec), flush=True)


# --------------------------------------------------------------------- prep
if args.mode == "prep" and args.synthetic:
    # CPU dry run of this driver (no download): same file shapes, fake text.
    words = ["the", "cat", "sat", "on", "the", "mat", "and", "the", "dog", "ran", "to", "the", "park"]
    rng = random.Random(0)
    fake = [{"messages": [{"role": "user", "content": " ".join(rng.choices(words, k=6))},
                          {"role": "assistant", "content": " ".join(rng.choices(words, k=8))}]}
            for _ in range(args.n_train + args.n_heldout)]
    for path, part in ((TRAIN, fake[: args.n_train]), (HELD, fake[args.n_train:])):
        with open(path, "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    dump(os.path.join(OUT, "prep.json"), {"mode": "prep", "dataset": "synthetic (dry run)"})
    raise SystemExit(0)

if args.mode == "prep":
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for ex in ds:
        user = ex["instruction"].strip()
        if ex.get("context"):
            user += "\n\n" + ex["context"].strip()
        resp = ex["response"].strip()
        if 20 <= len(resp) <= 1500 and len(user) <= 1500:
            rows.append({"messages": [{"role": "user", "content": user},
                                      {"role": "assistant", "content": resp}]})
    random.Random(0).shuffle(rows)
    train, held = rows[: args.n_train], rows[args.n_train: args.n_train + args.n_heldout]
    for path, part in ((TRAIN, train), (HELD, held)):
        with open(path, "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    dump(os.path.join(OUT, "prep.json"), {
        "mode": "prep", "dataset": "databricks/databricks-dolly-15k", "license": "CC BY-SA 3.0",
        "n_train": len(train), "n_heldout": len(held), "pool": len(rows), "shuffle_seed": 0,
        "disjoint": not ({json.dumps(r) for r in train} & {json.dumps(r) for r in held}),
    })
    raise SystemExit(0)


# ---------------------------------------------------------------- summarize
def load_runs(prefix: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(RUNS)):
        if name.startswith(prefix) and name.endswith(".json"):
            with open(os.path.join(RUNS, name)) as fh:
                out.append(json.load(fh))
    return out


def spread(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 4),
            "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


if args.mode == "summarize":
    stage = args.stage
    runs = load_runs(f"{stage}_")
    bases = {r["model"]: r for r in load_runs("base_")}
    checks: dict[str, bool] = {}
    table: dict[str, dict] = {}
    for r in runs:
        key = f'{r["model"]}|{r["arm"]}'
        table.setdefault(key, {"runs": []})["runs"].append(r)
    for entry in table.values():
        rs = entry.pop("runs")
        ok = [r for r in rs if r.get("status") == "ok"]
        entry["seeds"] = sorted(r["seed"] for r in rs)
        entry["status"] = sorted({r.get("status") for r in rs})
        entry["heldout_after"] = spread([r.get("heldout_after") for r in ok])
        entry["heldout_before"] = ok[0].get("heldout_before") if ok else None
        entry["delta"] = spread([r["heldout_after"] - r["heldout_before"] for r in ok
                                 if r.get("heldout_after") is not None and r.get("heldout_before") is not None])
        entry["s_per_step"] = spread([r.get("s_per_step") for r in ok])
        entry["peak_alloc_gib"] = spread([r.get("peak_vram_alloc_gib") for r in ok])
        entry["peak_reserved_gib"] = spread([r.get("peak_vram_reserved_gib") for r in ok])
        entry["projected_total_gb"] = ok[0].get("fit_projection", {}).get("total_gb") if ok else None
        entry["block_visits"] = sorted({(r.get("engine_summary") or {}).get("block_visits") for r in ok}, key=str)
    for model, b in bases.items():
        checks[f"base_in_bounds:{model}"] = bool(b.get("in_bounds"))
    if stage in ("a", "a2", "c"):
        for r in runs:
            checks[f'completed:{r["tag"]}'] = r.get("status") == "ok"
            checks[f'finite:{r["tag"]}'] = r.get("status") == "ok" and all(
                math.isfinite(x) for x in r.get("losses", []))
    if stage == "b":
        byarm = {r["arm"]: r for r in runs}
        b1 = byarm.get("block_uncapped", {})
        checks["b1_completed"] = b1.get("status") == "ok"
        checks["b1_two_switches"] = len((b1.get("engine_summary") or {}).get("switches", [])) >= 2
        checks["b1_loss_finite"] = bool(b1.get("losses")) and all(math.isfinite(x) for x in b1["losses"])
        checks["b1_generated"] = bool((b1.get("generation") or "").strip())
        checks["b1_saved_bf16"] = b1.get("saved_dtype") == "torch.bfloat16"
        checks["qlora_completed"] = byarm.get("qlora", {}).get("status") == "ok"
        for arm in ("block_cap24", "block_cap24_frozen_embed", "block_cap16_frozen_embed"):
            if arm in byarm:  # recorded, not gated: fitting or not is the measurement
                table.setdefault(f"cap|{arm}", {})["fits"] = byarm[arm].get("status") == "ok"
    rec = {"mode": "summarize", "stage": stage, "git_sha": args.git_sha, "table": table,
           "bases": {m: {k: b.get(k) for k in ("heldout_loss", "ceiling_ln_vocab", "in_bounds")}
                     for m, b in bases.items()},
           "checks": checks}
    dump(os.path.join(OUT, f"stage_{stage}.json"), rec)
    failed = [k for k, v in checks.items() if not v]
    print(f"RESULT stage {stage}: " + ("PASS" if checks and not failed else "FAIL " + ",".join(failed)), flush=True)
    raise SystemExit(0 if checks and not failed else 1)


# ----------------------------------------------------------- base / train
os.environ.setdefault("UNSLOTH_AUTO_INSTALL", "0")
os.environ["BACKPROPAGATE_TRAINING__SEED"] = str(args.seed)
os.environ["BACKPROPAGATE_TRAINING__LOGGING_STEPS"] = "1"
os.environ["BACKPROPAGATE_TRAINING__SAVE_STEPS"] = "1000000"

import importlib.metadata as md  # noqa: E402

import torch  # noqa: E402

torch.manual_seed(args.seed)
random.seed(args.seed)
# CUDA on the pod. Without CUDA the driver still runs end to end on CPU (a dry
# run of the driver itself, e.g. with a tiny local model); VRAM fields read 0.
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory if CUDA else 0
if args.vram_cap_gb and CUDA:
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * GIB / TOTAL_VRAM), 0)


def peak_alloc() -> float:
    return torch.cuda.max_memory_allocated() / GIB if CUDA else 0.0


def peak_reserved() -> float:
    return torch.cuda.max_memory_reserved() / GIB if CUDA else 0.0


def reset_peaks() -> None:
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def versions() -> dict:
    out = {}
    for p in ("torch", "transformers", "trl", "accelerate", "peft", "bitsandbytes", "datasets", "backpropagate"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def env() -> dict:
    return {"gpu": torch.cuda.get_device_name(0) if CUDA else "cpu (dry run)",
            "vram_total_gib": round(TOTAL_VRAM / GIB, 2),
            "cuda": torch.version.cuda, "versions": versions(), "git_sha": args.git_sha}


def heldout_texts() -> list[str]:
    from backpropagate.datasets import DatasetLoader

    return list(DatasetLoader(HELD, validate=False).to_hf_dataset()["text"])


@torch.no_grad()
def heldout_loss(model, tok) -> dict:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    dev = next(model.parameters()).device
    for text in heldout_texts():
        ids = tok(text, truncation=True, max_length=args.seq, add_special_tokens=False,
                  return_tensors="pt")["input_ids"].to(dev)
        if ids.shape[-1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n = ids.shape[-1] - 1
        total += float(out.loss) * n
        count += n
    if was_training:
        model.train()
    return {"loss": round(total / count, 4), "tokens": count}


if args.mode == "base":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEVICE)
    h = heldout_loss(m, tok)
    ceiling = math.log(m.config.vocab_size)
    rec = {"mode": "base", "model": args.model, "heldout_loss": h["loss"], "heldout_tokens": h["tokens"],
           "floor": 0.0, "ceiling_ln_vocab": round(ceiling, 4), "in_bounds": 0.0 < h["loss"] < ceiling,
           "params": sum(p.numel() for p in m.parameters()), **env()}
    dump(os.path.join(RUNS, f"base_{args.tag or slug(args.model)}.json"), rec)
    raise SystemExit(0 if rec["in_bounds"] else 1)


# ------------------------------------------------------------------- train
from backpropagate import block_engine as be  # noqa: E402
from backpropagate.trainer import Trainer, TrainingCallback  # noqa: E402

tag = args.tag or f"run_{int(time.time())}"
arm = ("qlora" if args.qlora else args.engine)
rec: dict = {"mode": "train", "tag": tag, "arm": os.environ.get("ARM", arm), "model": args.model,
             "engine": "qlora" if args.qlora else args.engine, "seed": args.seed, "steps": args.steps,
             "batch": args.batch, "seq": args.seq, "lr": args.lr, "k": args.k, "order": args.order,
             "writeback": args.writeback, "freeze_embeddings": args.freeze_embeddings,
             "vram_cap_gib": args.vram_cap_gb or None, **env()}
base_path = os.path.join(RUNS, f"base_{slug(args.model)}.json")
if os.path.exists(base_path):
    with open(base_path) as fh:
        rec["heldout_before"] = json.load(fh)["heldout_loss"]

# Per-block-visit peak VRAM: wrap the engine's deactivate (the end of a visit).
visit_peaks: list[dict] = []
_orig_deactivate = be.BlockCoordinateOptimizer._deactivate


def _deactivate_with_peak(self):  # type: ignore[no-untyped-def]
    blk = self.active_block
    if blk is not None:
        visit_peaks.append({"block": blk.name, "params": blk.numel, "steps": self.steps_in_block,
                            "peak_alloc_gib": round(peak_alloc(), 3),
                            "peak_reserved_gib": round(peak_reserved(), 3)})
        reset_peaks()
    return _orig_deactivate(self)


be.BlockCoordinateOptimizer._deactivate = _deactivate_with_peak  # type: ignore[method-assign]

kwargs: dict = {"model": args.model, "use_unsloth": False, "max_seq_length": args.seq, "batch_size": args.batch,
                    "gradient_accumulation": 1, "packing": False, "learning_rate": args.lr,
                    "output_dir": os.path.join(WORKDIR, tag), "report_to": "none"}
if args.qlora:
    kwargs.update(mode="lora", learning_rate=2e-4 if args.lr == 2e-5 else args.lr)
else:
    kwargs["mode"] = "full"
    if args.engine == "block":
        kwargs.update(full_ft_engine="block", switch_block_every=args.k, block_order=args.order,
                      block_writeback=args.writeback, block_train_embeddings=not args.freeze_embeddings)
if args.vram_cap_gb:
    kwargs["oom_recovery"] = False
# Escape hatch for the operator (JSON merged into Trainer(...)), as in pod_offload_7b.sh.
kwargs.update(json.loads(os.environ.get("BP_TRAINER_KWARGS", "{}") or "{}"))  # measure the fit; do not silently halve the batch

stamps: list[float] = []
losses: list[float] = []


def on_step(step, loss):  # type: ignore[no-untyped-def]  # noqa: ARG001
    stamps.append(time.perf_counter())
    losses.append(round(float(loss), 4))


t0 = time.perf_counter()
peak_before_error = None
try:
    t = Trainer(**kwargs)
    t.load_model()
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    rec["params"] = sum(p.numel() for p in t._model.parameters())
    if not args.qlora and args.engine == "block":
        fit = be.fit_from_model(t._model, args.seq, args.batch, include_embeddings=not args.freeze_embeddings)
        rec["fit_projection"] = fit.as_dict()
        rec["partition"] = be.partition_model(t._model, include_embeddings=not args.freeze_embeddings).summary()
    reset_peaks()
    t1 = time.perf_counter()
    run = t.train(TRAIN, steps=args.steps, callback=TrainingCallback(on_step=on_step))
    rec["train_s"] = round(time.perf_counter() - t1, 1)
    # Drop the trainer's end-of-run checkpoint now: disk, not evidence.
    shutil.rmtree(os.path.join(WORKDIR, tag), ignore_errors=True)
    rec["peak_vram_alloc_gib"] = round(max([peak_alloc()]
                                           + [v["peak_alloc_gib"] for v in visit_peaks]), 3)
    rec["peak_vram_reserved_gib"] = round(max([peak_reserved()]
                                              + [v["peak_reserved_gib"] for v in visit_peaks]), 3)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    rec["s_per_step"] = round(statistics.median(gaps), 4) if gaps else None
    rec["s_per_step_source"] = "median gap between per-step log callbacks"
    rec["losses"] = losses or [round(x, 4) for x in run.loss_history]
    rec["final_loss"] = run.final_loss
    rec["engine_summary"] = run.metadata.get("block_engine")
    rec["visit_peaks"] = visit_peaks
    if args.gen_inprocess:
        m, tok = t._model, t._tokenizer
        m.eval()
        ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(next(m.parameters()).device)
        with torch.no_grad():
            out = m.generate(input_ids=ids, max_new_tokens=24, do_sample=False)
        rec["generation"] = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        rec["model_class"] = type(m).__name__
    if not args.no_eval:
        rec["heldout_after"] = heldout_loss(t._model, t._tokenizer)["loss"]
    if args.save_reload:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        save_dir = os.path.join(SAVEDIR, f"saved_{tag}")
        saved = t.save(save_dir, run_id=run.run_id)
        del t, run
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(saved)
        m = AutoModelForCausalLM.from_pretrained(saved, dtype=torch.bfloat16, device_map=DEVICE)
        ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(DEVICE)
        out = m.generate(ids, max_new_tokens=24, do_sample=False)
        rec["generation"] = tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        rec["saved_dtype"] = str(next(m.parameters()).dtype)
        rec["reloaded_heldout"] = heldout_loss(m, tok)["loss"]
        del m
        shutil.rmtree(saved, ignore_errors=True)
    rec["status"] = "ok"
except Exception as exc:  # a capped run that does not fit is a result, not a crash
    msg = f"{type(exc).__name__}: {getattr(exc, 'code', '')}: {exc}"
    cause = f"{exc.__cause__!r}" if exc.__cause__ is not None else ""
    oom = any(s in (msg + cause) for s in ("out of memory", "OutOfMemory", "RUNTIME_GPU_OOM",
                                             "RUNTIME_OOM"))
    rec["status"] = "oom" if oom else "error"
    rec["error"] = msg[:2000]
    rec["traceback_tail"] = traceback.format_exc()[-3000:]
    rec["peak_vram_alloc_gib_before_error"] = round(max([peak_alloc()]
                                                        + [v["peak_alloc_gib"] for v in visit_peaks]), 3)
    rec["visit_peaks"] = visit_peaks
    rec["losses"] = losses
finally:
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    shutil.rmtree(os.path.join(WORKDIR, tag), ignore_errors=True)

dump(os.path.join(RUNS, f"{tag}.json"), rec)
# exit 0 for ok and for an OOM under a cap (a measured "does not fit"); 1 otherwise
sys.exit(0 if rec["status"] == "ok" or (rec["status"] == "oom" and args.vram_cap_gb) else 1)
