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
    inspect    load a model with the library's text-only loader and count what
               loaded (params, vision-tower tensors) against the Hub checkpoint

``--dataset gsm8k`` (stage d) switches prep/base/train to GSM8K (openai/gsm8k,
config "main", MIT): train on the shuffled train split, evaluate on a fixed
250-question sample of the test split with two metrics — held-out loss on the
answer tokens only, and accuracy = the number after ``####`` in a greedy
generation equals the gold number (a lenient last-number accuracy is recorded
next to it, to separate format failures from arithmetic). Every arm sees the
same prompt: the library's ChatML text (DatasetLoader), user turn = question +
GSM_INSTRUCTION, generation prompted with ``<|im_start|>assistant\n``.

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
ap.add_argument("mode", choices=["prep", "base", "train", "summarize", "inspect", "lengths"])
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
ap.add_argument("--lr-default", action="store_true",
                help="pass no learning_rate: each arm uses the library's documented default")
ap.add_argument("--dataset", choices=["dolly", "gsm8k"], default="dolly")
ap.add_argument("--eval-batch", type=int, default=50)
ap.add_argument("--max-new-tokens", type=int, default=400)
ap.add_argument("--n-test", type=int, default=250)
ap.add_argument("--optim", default=None, help="train: Trainer(optim=...), e.g. adafactor")
ap.add_argument("--galore", default=None,
                help="train: GaLore optim name (galore_adamw_8bit_layerwise | galore_adamw_8bit)")
ap.add_argument("--galore-args", default="rank=128, update_proj_gap=200, scale=0.25")
ap.add_argument("--ceiling", type=float, default=None, help="train: full_ft_ceiling_billions override")
ap.add_argument("--library-defaults", action="store_true",
                help="train: pass only model/mode/output (batch auto, library seq/packing/lr)")
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
GOLD = os.path.join(OUT, "gsm8k_test_gold.json")
if args.dataset == "gsm8k":
    TRAIN = os.path.join(OUT, "gsm8k_train.jsonl")
    HELD = os.path.join(OUT, "gsm8k_test.jsonl")
GSM_INSTRUCTION = ("Solve the math word problem. Show your reasoning step by step, then give the "
                   "final numeric answer on its own last line in the form '#### <number>'.")
ASSISTANT_MARKER = "<|im_start|>assistant\n"
GIB = 2**30
# Trainer output (its end-of-run checkpoint) and the save->reload copy can live
# on fast local disk while receipts stay on the (network) volume.
WORKDIR = os.environ.get("BP_WORK_DIR") or os.path.join(OUT, "work")
SAVEDIR = os.environ.get("BP_SAVE_DIR") or WORKDIR


def slug(model: str) -> str:
    """File-name form of a model id ("Qwen/Qwen2.5-7B" -> "Qwen_Qwen2.5-7B")."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in model)


def dump(path: str, rec: dict) -> None:
    if "manifest" in rec:  # run manifest first in every receipt
        rec = {"manifest": rec.pop("manifest"), **rec}
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

def gsm_number(text: str) -> float | None:
    t = text.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


if args.mode == "prep" and args.dataset == "gsm8k":
    import re

    from datasets import load_dataset

    def row(q: str, a: str) -> dict:
        # GSM8K answers carry calculator annotations "<<48/2=24>>"; strip them.
        return {"messages": [{"role": "user", "content": f"{q.strip()}\n\n{GSM_INSTRUCTION}"},
                             {"role": "assistant", "content": re.sub(r"<<[^>]*>>", "", a).strip()}]}

    tr = load_dataset("openai/gsm8k", "main", split="train")
    te = load_dataset("openai/gsm8k", "main", split="test")
    train_rows = [row(ex["question"], ex["answer"]) for ex in tr]
    random.Random(0).shuffle(train_rows)
    idx = sorted(random.Random(0).sample(range(len(te)), args.n_test))
    test_rows = [row(te[i]["question"], te[i]["answer"]) for i in idx]
    gold = [gsm_number(te[i]["answer"].split("####")[-1]) for i in idx]
    for path, part in ((TRAIN, train_rows), (HELD, test_rows)):
        with open(path, "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    with open(GOLD, "w") as fh:
        json.dump(gold, fh)
    dump(os.path.join(OUT, "prep_gsm8k.json"), {
        "mode": "prep", "dataset": "openai/gsm8k (main)", "license": "MIT",
        "n_train": len(train_rows), "n_test": len(test_rows), "test_indices": idx,
        "shuffle_seed": 0, "gold_parsed": sum(g is not None for g in gold),
        "prompt_format": {
            "text": "library ChatML via DatasetLoader: <|im_start|>user\\n{question}\\n\\n{instruction}"
                    "<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>",
            "instruction": GSM_INSTRUCTION,
            "answer": "GSM8K solution with <<...>> calculator annotations removed, ending '#### N'",
            "generation": "greedy, prompt ends at '<|im_start|>assistant\\n', stop on <|im_end|>/eos",
        },
    })
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
        if any("acc_strict" in r for r in ok):
            entry["acc_strict"] = spread([r.get("acc_strict") for r in ok])
            entry["acc_lenient"] = spread([r.get("acc_lenient") for r in ok])
            entry["lr_used"] = sorted({r.get("lr_used") for r in ok}, key=str)
            entry["eval_s"] = spread([r.get("eval_s") for r in ok])
    for model, b in bases.items():
        checks[f"base_in_bounds:{model}"] = bool(b.get("in_bounds"))
    if stage == "d":
        bases = {m: b for m, b in bases.items() if b.get("dataset") == "gsm8k"}
        checks = {f"base_acc_in_bounds:{m}": bool(b.get("in_bounds")) for m, b in bases.items()}
    if stage in ("a", "a2", "c", "d"):
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
           "bases": {m: {k: b.get(k) for k in ("heldout_loss", "ceiling_ln_vocab", "in_bounds",
                                               "acc_strict", "acc_lenient")}
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
CARD = torch.cuda.get_device_name(0) if CUDA else "cpu (dry run)"
# 5090 emulation on a bigger card (e.g. RTX PRO 6000 Blackwell, 96 GB): cap
# every process at the 5090's usable total and make the library see a 5090-
# sized device, so auto batch size, full-FT ceilings and optimizer choice
# resolve exactly as on the 5090. NVML peaks are system-wide and ignore the
# cap; torch reserved under the cap is the figure comparable to 5090 runs.
RTX5090_USABLE_GIB = 31.36
VRAM_CAP_GIB = args.vram_cap_gb or None
EMULATED_5090 = False
if CUDA and not args.vram_cap_gb and TOTAL_VRAM / GIB > 33.0:
    VRAM_CAP_GIB = RTX5090_USABLE_GIB
    EMULATED_5090 = True
if CUDA and VRAM_CAP_GIB:
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GIB * GIB / TOTAL_VRAM), 0)
if EMULATED_5090:
    _real_props = torch.cuda.get_device_properties

    class _CappedProps:
        def __init__(self, props, total):  # type: ignore[no-untyped-def]
            self._props, self.total_memory = props, total

        def __getattr__(self, name):  # type: ignore[no-untyped-def]
            return getattr(self._props, name)

    def _capped_get_device_properties(device=None):  # type: ignore[no-untyped-def]
        props = _real_props(0 if device is None else device)
        return _CappedProps(props, int(RTX5090_USABLE_GIB * GIB))

    torch.cuda.get_device_properties = _capped_get_device_properties  # type: ignore[assignment]
SPEED_LABEL = f"measured on {CARD}" + (" (capped at 31.36 GiB to emulate an RTX 5090)" if EMULATED_5090 else "")


class NvmlSampler:
    """System-wide device memory (NVML), sampled at ~10 Hz on a thread.

    torch's allocator counters miss memory allocated outside it — notably
    bitsandbytes' paged optimizer state (CUDA managed memory via
    ``cget_managed_ptr``). NVML sees everything resident on the device,
    including the CUDA context. ``mark(phase)`` starts a named phase whose peak
    is tracked separately (e.g. load / train / eval).
    """

    def __init__(self) -> None:
        import threading

        self.ok = False
        self.peak = 0
        self.phase = "start"
        self.phase_peak: dict[str, int] = {}
        self.baseline = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.driver = pynvml.nvmlSystemGetDriverVersion()
            self.baseline = self._used()
            self.ok = True
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            self.error = f"{type(exc).__name__}: {exc}"
            self.driver = None
            return
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _used(self) -> int:
        return int(self._nv.nvmlDeviceGetMemoryInfo(self._h).used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                u = self._used()
            except Exception:  # noqa: BLE001  # nosec B112 - sampling is best effort
                continue
            self.peak = max(self.peak, u)
            self.phase_peak[self.phase] = max(self.phase_peak.get(self.phase, 0), u)
            self._stop.wait(0.1)

    def mark(self, phase: str) -> None:
        self.phase = phase

    def report(self) -> dict:
        if not self.ok:
            return {"nvml": "unavailable", "nvml_error": getattr(self, "error", None)}
        return {"nvml_peak_gib": round(self.peak / GIB, 3),
                "nvml_baseline_gib": round((self.baseline or 0) / GIB, 3),
                "nvml_phase_peak_gib": {k: round(v / GIB, 3) for k, v in self.phase_peak.items()},
                "nvml_sample_hz": 10}


NVML = NvmlSampler() if CUDA else None


def nvml_report() -> dict:
    return NVML.report() if NVML is not None else {"nvml": "no cuda"}


def nvml_mark(phase: str) -> None:
    if NVML is not None:
        NVML.mark(phase)


def peak_alloc() -> float:
    return torch.cuda.max_memory_allocated() / GIB if CUDA else 0.0


def peak_reserved() -> float:
    return torch.cuda.max_memory_reserved() / GIB if CUDA else 0.0


def reset_peaks() -> None:
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def versions() -> dict:
    out = {}
    for p in ("torch", "transformers", "trl", "accelerate", "peft", "bitsandbytes", "galore-torch",
              "nvidia-ml-py", "unsloth", "datasets", "backpropagate"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def env() -> dict:
    return {"manifest": {
        "git_sha": args.git_sha, "image": os.environ.get("POD_IMAGE", "unknown"),
        "gpu": torch.cuda.get_device_name(0) if CUDA else "cpu (dry run)",
        "vram_total_gib": round(TOTAL_VRAM / GIB, 2), "cuda_runtime": torch.version.cuda,
        "card": CARD, "card_total_gib": round(TOTAL_VRAM / GIB, 2), "vram_cap_gib": VRAM_CAP_GIB,
        "emulated_rtx5090": EMULATED_5090,
        "comparable_vram_figure": ("torch max reserved under the cap (NVML is system-wide and ignores it)"
                                   if EMULATED_5090 else "NVML peak and torch max reserved"),
        "cuda_driver": getattr(NVML, "driver", None), "versions": versions(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }, "gpu": torch.cuda.get_device_name(0) if CUDA else "cpu (dry run)",
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


def _extract_strict(text: str) -> float | None:
    import re

    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    return gsm_number(m[-1]) if m else None


def _extract_lenient(text: str) -> float | None:
    import re

    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return gsm_number(m[-1]) if m else None


def wilson(p: float, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-6


@torch.no_grad()
def gsm_eval(model, tok) -> dict:
    """Answer-token held-out loss + greedy-generation accuracy on the test sample."""
    t_start = time.perf_counter()
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    texts = heldout_texts()
    with open(GOLD) as fh:
        gold = json.load(fh)
    qids = list(range(len(gold)))
    _prep = os.path.join(OUT, "prep_gsm8k.json")
    if os.path.exists(_prep):
        with open(_prep) as fh:
            qids = json.load(fh).get("test_indices", qids)
    prompts, answers = [], []
    for text in texts:
        i = text.rindex(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
        prompts.append(text[:i])
        answers.append(text[i:])
    total, count, prefix_mismatch = 0.0, 0, 0
    item_loss: list[tuple[float, int]] = []
    for pr, an in zip(prompts, answers):
        pi = tok(pr, add_special_tokens=False)["input_ids"]
        fi = tok(pr + an, add_special_tokens=False, truncation=True, max_length=args.seq)["input_ids"]
        prefix_mismatch += int(fi[: len(pi)] != pi)
        ids = torch.tensor([fi], device=dev)
        labels = ids.clone()
        labels[:, : len(pi)] = -100
        n = int((labels[:, 1:] != -100).sum())
        if n < 1:
            item_loss.append((0.0, 0))
            continue
        out = model(input_ids=ids, labels=labels)
        total += float(out.loss) * n
        count += n
        item_loss.append((float(out.loss) * n, n))
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id and im_end not in stop:
        stop.append(im_end)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[str] = [""] * len(prompts)
    hit_limit = 0
    for s0 in range(0, len(order), args.eval_batch):
        idx = order[s0: s0 + args.eval_batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=stop, pad_token_id=tok.pad_token_id, use_cache=True)
        new = gen[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            outs[i] = tok.decode(new[j], skip_special_tokens=True)
            hit_limit += int(not any(int(t) in stop for t in new[j]))
    tok.padding_side = side
    strict = [_same(_extract_strict(o), g) for o, g in zip(outs, gold)]
    items = [{"qid": qids[i], "gold": gold[i], "generated": outs[i], "parsed": _extract_strict(outs[i]),
              "correct": strict[i], "answer_loss_sum": round(item_loss[i][0], 6),
              "answer_tokens": item_loss[i][1]} for i in range(len(outs))]
    lenient = [_same(_extract_lenient(o), g) for o, g in zip(outs, gold)]
    if was_training:
        model.train()
    return {
        "answer_loss": round(total / count, 4) if count else None, "answer_tokens": count,
        "acc_strict": round(sum(strict) / len(strict), 4),
        "acc_lenient": round(sum(lenient) / len(lenient), 4),
        "has_hash_answer": round(sum(_extract_strict(o) is not None for o in outs) / len(outs), 4),
        "hit_max_new_tokens": hit_limit, "prefix_mismatch": prefix_mismatch,
        "n": len(outs), "eval_s": round(time.perf_counter() - t_start, 1),
        "samples": [{"output": outs[i][:600], "gold": gold[i], "strict": strict[i]} for i in range(3)],
        "items": items,
        "template_example_prompt": prompts[0],
    }


def write_items(tag: str, ev: dict) -> str:
    """Per-item outputs for the paired tests: runs/items/<tag>.jsonl."""
    d = os.path.join(RUNS, "items")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{tag}.jsonl")
    with open(path, "w") as fh:
        for it in ev["items"]:
            fh.write(json.dumps(it) + "\n")
    return os.path.relpath(path, OUT)


if args.mode == "lengths":
    # Token lengths of the exact training texts, to choose max_seq_length: how
    # many examples would lose their gold '#### N' line at 512 / 768 tokens?
    from transformers import AutoTokenizer

    from backpropagate.datasets import DatasetLoader

    tok = AutoTokenizer.from_pretrained(args.model)
    rec = {"mode": "lengths", "model": args.model, **env()}
    for name, path in (("train", TRAIN), ("test", HELD)):
        texts = list(DatasetLoader(path, validate=False).to_hf_dataset()["text"])
        full = [len(tok(t_, add_special_tokens=False)["input_ids"]) for t_ in texts]
        # position of the end of the '#### N' line in tokens
        ends = []
        for t_ in texts:
            j = t_.rfind("####")
            k = t_.find("<|im_end|>", j)
            ends.append(len(tok(t_[: k if k > 0 else len(t_)], add_special_tokens=False)["input_ids"]))
        full_sorted = sorted(full)

        def pct(q: float, xs: list[int] = full_sorted) -> int:
            return xs[min(len(xs) - 1, int(q * len(xs)))]

        rec[name] = {"n": len(full), "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": max(full),
                     "frac_gold_truncated_at_512": round(sum(e > 512 for e in ends) / len(ends), 4),
                     "frac_gold_truncated_at_768": round(sum(e > 768 for e in ends) / len(ends), 4),
                     "frac_text_over_512": round(sum(f > 512 for f in full) / len(full), 4)}
    worst = max(rec["train"]["frac_gold_truncated_at_512"], rec["test"]["frac_gold_truncated_at_512"])
    rec["seq_choice"] = 768 if worst > 0.01 else 512
    rec["rule"] = "768 if the gold '####' line is truncated at 512 for > 1% of train or test examples"
    rec["dropped"] = "nothing: long examples are truncated by the trainer, not filtered"
    dump(os.path.join(RUNS, f"lengths_{slug(args.model)}.json"), rec)
    raise SystemExit(0)


if args.mode == "inspect":
    # What does the library's text-only loader (AutoModelForCausalLM) load from
    # this checkpoint, against what the checkpoint holds?
    from huggingface_hub import hf_hub_download, model_info
    from transformers import AutoModelForCausalLM

    info = model_info(args.model, files_metadata=False)
    st = getattr(info, "safetensors", None)
    rec = {"mode": "inspect", "model": args.model, "pipeline_tag": info.pipeline_tag,
           "hub_safetensors_total": getattr(st, "total", None),
           "hub_safetensors_by_dtype": dict(getattr(st, "parameters", {}) or {}), **env()}
    try:
        with open(hf_hub_download(args.model, "model.safetensors.index.json")) as fh:
            keys = list(json.load(fh)["weight_map"])
        rec["checkpoint_tensors"] = len(keys)
        rec["checkpoint_vision_tensors"] = sum(("visual" in k or "vision" in k) for k in keys)
    except Exception as exc:  # noqa: BLE001 - single-file checkpoints have no index
        rec["checkpoint_index_error"] = f"{type(exc).__name__}: {exc}"[:300]
    reset_peaks()
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEVICE)
    names = [n for n, _ in m.named_parameters()]
    rec["loaded_class"] = type(m).__name__
    rec["loaded_params"] = sum(p.numel() for p in m.parameters())
    rec["loaded_vision_params"] = sum(p.numel() for n, p in m.named_parameters()
                                      if "visual" in n or "vision" in n)
    rec["loaded_tensors"] = len(names)
    rec["peak_vram_alloc_gib_bf16_load"] = round(peak_alloc(), 3)
    rec.update(nvml_report())
    dump(os.path.join(RUNS, f"{args.tag or 'inspect_' + slug(args.model)}.json"), rec)
    raise SystemExit(0)


if args.mode == "base" and args.dataset == "gsm8k":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEVICE)
    ev = gsm_eval(m, tok)
    ev["items_file"] = write_items(f"base_gsm8k_{args.tag or slug(args.model)}", ev)
    ev.pop("items")
    acc = ev["acc_strict"]
    rec = {"mode": "base", "dataset": "gsm8k", "model": args.model, "heldout_loss": ev["answer_loss"],
           **ev, "floor": 0.0, "ceiling": 1.0, "in_bounds": 0.0 < acc < 1.0,
           "near_ceiling": acc >= 0.9, "near_floor": acc <= 0.02,
           "acc_wilson95": wilson(acc, ev["n"]), **env(), **nvml_report()}
    dump(os.path.join(RUNS, f"base_gsm8k_{args.tag or slug(args.model)}.json"), rec)
    raise SystemExit(0 if rec["in_bounds"] else 1)


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
from backpropagate.config import settings as bp_settings  # noqa: E402
from backpropagate.trainer import Trainer, TrainingCallback  # noqa: E402

# Without pydantic-settings installed the library's settings ignore the
# BACKPROPAGATE_* env vars set above (measured on the pod: seed stayed 42), so
# set the three knobs this driver depends on directly as well.
bp_settings.training.seed = args.seed
bp_settings.training.logging_steps = 1
bp_settings.training.save_steps = 1_000_000

# No trainer checkpoints in these measurement runs: HF saves one at the last
# step whatever save_steps is, and a 7B one (15 GB model + optimizer) filled the
# pod's 40 GB container disk in the first stage-b attempt. Checkpoint/resume is
# covered by the unit tests; save->reload here goes through Trainer.save().
_orig_build_args = Trainer._build_training_args


def _build_args_no_checkpoints(self, **kw):  # type: ignore[no-untyped-def]
    cfg = _orig_build_args(self, **kw)
    from transformers.trainer_utils import SaveStrategy

    cfg.save_strategy = SaveStrategy.NO
    if args.galore:
        # GaLore (Zhao et al. 2024): transformers' built-in optimizer. Target the
        # attention and MLP linears; rank / update gap / scale from --galore-args.
        cfg.optim_target_modules = ["attn", "mlp"]
        cfg.optim_args = args.galore_args
    return cfg


Trainer._build_training_args = _build_args_no_checkpoints  # type: ignore[method-assign]

tag = args.tag or f"run_{int(time.time())}"
arm = ("qlora" if args.qlora else args.engine)
rec: dict = {"mode": "train", "tag": tag, "arm": os.environ.get("ARM", arm), "model": args.model,
             "engine": "qlora" if args.qlora else args.engine, "seed": args.seed, "steps": args.steps,
             "batch": args.batch, "seq": args.seq, "lr": args.lr, "k": args.k, "order": args.order,
             "writeback": args.writeback, "freeze_embeddings": args.freeze_embeddings,
             "vram_cap_gib": VRAM_CAP_GIB, **env(),
             "settings_seed": bp_settings.training.seed,
             "settings_logging_steps": bp_settings.training.logging_steps}
rec["dataset"] = args.dataset
base_path = os.path.join(RUNS, f"base_{'gsm8k_' if args.dataset == 'gsm8k' else ''}{slug(args.model)}.json")
if os.path.exists(base_path):
    with open(base_path) as fh:
        _b = json.load(fh)
    rec["heldout_before"] = _b["heldout_loss"]
    rec["acc_before"] = _b.get("acc_strict")

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
if args.lr_default:
    kwargs.pop("learning_rate")
if args.optim:
    kwargs["optim"] = args.optim
if args.galore:
    kwargs["optim"] = args.galore
if args.ceiling:
    kwargs["full_ft_ceiling_billions"] = args.ceiling
if args.library_defaults:
    # Library defaults: batch "auto", settings' seq length / packing / LR.
    kwargs = {"model": args.model, "mode": "lora" if args.qlora else "full",
              "output_dir": os.path.join(WORKDIR, tag), "report_to": "none"}
if args.vram_cap_gb:
    kwargs["oom_recovery"] = False  # measure the fit; do not silently halve the batch
# Escape hatch for the operator (JSON merged into Trainer(...)), as in pod_offload_7b.sh.
kwargs.update(json.loads(os.environ.get("BP_TRAINER_KWARGS", "{}") or "{}"))
rec["trainer_kwargs"] = {k: v for k, v in kwargs.items() if k != "output_dir"}

stamps: list[float] = []
losses: list[float] = []


def visit_log(peaks: list[dict], step_losses: list[float]) -> list[dict]:
    """Per block visit: block, global step range, train loss at entry / exit."""
    out, start = [], 1
    for v in peaks:
        end = start + v["steps"] - 1
        entry = step_losses[start - 1] if 0 < start <= len(step_losses) else None
        exit_ = step_losses[end - 1] if 0 < end <= len(step_losses) and v["steps"] else None
        out.append({"block": v["block"], "steps": [start, end] if v["steps"] else None,
                    "loss_entry": entry, "loss_exit": exit_,
                    "peak_alloc_gib": v["peak_alloc_gib"], "peak_reserved_gib": v["peak_reserved_gib"]})
        start = end + 1
    return out


def describe_optimizer(t) -> dict:  # type: ignore[no-untyped-def]
    """Which optimizer actually ran (class + hyperparameters), and for
    bitsandbytes the size of its paged (managed-memory) state, which torch's
    counters cannot see."""
    out: dict = {}
    tr = getattr(t, "_trainer", None)
    opt = getattr(tr, "optimizer", None)
    inner = getattr(opt, "optimizer", opt)
    if inner is None:
        return {"optimizer": None}
    out["optimizer_class"] = f"{type(inner).__module__}.{type(inner).__name__}"
    try:
        out["optimizer_defaults"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                                     for k, v in dict(getattr(inner, "defaults", {})).items()}
    except Exception:  # noqa: BLE001
        out["optimizer_defaults"] = None
    paged = total = 0
    try:
        for st in inner.state.values():
            for v in (st.values() if isinstance(st, dict) else []):
                if torch.is_tensor(v):
                    b = v.numel() * v.element_size()
                    total += b
                    if getattr(v, "is_paged", False):
                        paged += b
    except Exception:  # noqa: BLE001
        pass
    out["optimizer_state_gib"] = round(total / GIB, 3)
    out["optimizer_paged_state_gib"] = round(paged / GIB, 3)
    sched = getattr(tr, "lr_scheduler", None)
    out["lr_scheduler_class"] = type(sched).__name__ if sched is not None else None
    out["max_grad_norm"] = getattr(getattr(tr, "args", None), "max_grad_norm", None)
    return out


FIRST_BATCHES: list[dict] = []


def _capture_training_step() -> None:
    """Record the first two batches the trainer actually trains on (input ids
    + which tokens carry loss). Same seed -> the same hashes in every arm, so
    this checks both identical loss masking and identical data order."""
    import hashlib

    from transformers import Trainer as HFTrainer

    orig = HFTrainer.training_step

    def training_step(self, model, inputs, *a, **kw):  # type: ignore[no-untyped-def]
        if len(FIRST_BATCHES) < 2:
            try:
                ids = inputs["input_ids"].detach().cpu()
                labels = inputs.get("labels")
                labels = labels.detach().cpu() if labels is not None else ids
                mask = labels != -100
                if "attention_mask" in inputs:
                    mask &= inputs["attention_mask"].detach().cpu().bool()
                FIRST_BATCHES.append({
                    "input_tokens": int(ids.numel()), "loss_tokens": int(mask.sum()),
                    "input_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:16],
                    "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest()[:16],
                    "shape": list(ids.shape)})
            except Exception as exc:  # noqa: BLE001
                FIRST_BATCHES.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
        return orig(self, model, inputs, *a, **kw)

    HFTrainer.training_step = training_step  # type: ignore[method-assign]


def mask_check(t) -> dict:  # type: ignore[no-untyped-def]  # noqa: ARG001
    if not FIRST_BATCHES:
        return {"error": "no batch captured"}
    out = dict(FIRST_BATCHES[0])
    if len(FIRST_BATCHES) > 1 and "input_sha256" in FIRST_BATCHES[1]:
        out["second_batch_input_sha256"] = FIRST_BATCHES[1]["input_sha256"]
    return out


_capture_training_step()


def on_step(step, loss):  # type: ignore[no-untyped-def]  # noqa: ARG001
    stamps.append(time.perf_counter())
    losses.append(round(float(loss), 4))


t0 = time.perf_counter()
peak_before_error = None
try:
    t = Trainer(**kwargs)
    rec["lr_used"] = t.learning_rate
    rec["batch_resolved"] = t.batch_size
    rec["max_seq_length_used"] = t.max_seq_length
    rec["packing_used"] = t.packing
    rec["use_unsloth_used"] = t.use_unsloth
    rec["optim_setting"] = t.optim
    rec["schedule"] = {"lr_scheduler_type": bp_settings.training.lr_scheduler_type,
                       "warmup_steps": bp_settings.training.warmup_steps,
                       "weight_decay": bp_settings.training.weight_decay,
                       "gradient_accumulation": t.gradient_accumulation}
    nvml_mark("load")
    rec["lora_r_used"] = t.lora_r if getattr(t, "mode", "lora") == "lora" else None
    t.load_model()
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    rec["params"] = sum(p.numel() for p in t._model.parameters())
    if not args.qlora and args.engine == "block":
        fit = be.fit_from_model(t._model, args.seq, args.batch, include_embeddings=not args.freeze_embeddings)
        rec["fit_projection"] = fit.as_dict()
        rec["partition"] = be.partition_model(t._model, include_embeddings=not args.freeze_embeddings).summary()
    if not args.qlora and args.engine == "block":
        rec["partition"]["blocks"] = be.partition_model(
            t._model, include_embeddings=not args.freeze_embeddings).names
    reset_peaks()
    nvml_mark("train")
    t1 = time.perf_counter()
    # The library caps the training set at settings.data.max_samples (default
    # 1000) unless samples= is passed. Found on the GSM8K pod run: 1000 steps x
    # batch 4 then meant 4 epochs over the first 1000 rows, not 0.53 epoch over
    # 7,473. Pass the full row count explicitly and record what was used.
    with open(TRAIN) as fh:
        n_rows = sum(1 for line in fh if line.strip())
    rec["train_rows_available"] = n_rows
    rec["train_samples_used"] = n_rows if args.dataset == "gsm8k" else min(n_rows, bp_settings.data.max_samples)
    rec["epochs_seen"] = round(args.steps * args.batch / rec["train_samples_used"], 3)
    run = t.train(TRAIN, steps=args.steps, samples=n_rows if args.dataset == "gsm8k" else None,
                  callback=TrainingCallback(on_step=on_step))
    nvml_mark("after_train")
    rec["torch_max_allocated_train_gib"] = round(peak_alloc(), 3)
    rec["torch_max_reserved_train_gib"] = round(peak_reserved(), 3)
    rec.update(describe_optimizer(t))
    rec["mask_check"] = mask_check(t)
    rec["train_s"] = round(time.perf_counter() - t1, 1)
    rec["time_to_first_step_s"] = round(stamps[0] - t1, 1) if stamps else None
    rec["oom_retries"] = run.metadata.get("oom_retries")
    rec["effective_batch_size"] = run.metadata.get("effective_batch_size")
    # Drop the trainer's end-of-run checkpoint now: disk, not evidence.
    shutil.rmtree(os.path.join(WORKDIR, tag), ignore_errors=True)
    rec["peak_vram_alloc_gib"] = round(max([peak_alloc()]
                                           + [v["peak_alloc_gib"] for v in visit_peaks]), 3)
    rec["peak_vram_reserved_gib"] = round(max([peak_reserved()]
                                              + [v["peak_reserved_gib"] for v in visit_peaks]), 3)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    rec["s_per_step"] = round(statistics.median(gaps), 4) if gaps else None
    rec["s_per_step_source"] = "median gap between per-step log callbacks"
    rec["s_per_step_measured_on"] = SPEED_LABEL
    rec["losses"] = losses or [round(x, 4) for x in run.loss_history]
    rec["final_loss"] = run.final_loss
    rec["engine_summary"] = run.metadata.get("block_engine")
    rec["visit_peaks"] = visit_peaks
    rec["visit_log"] = visit_log(visit_peaks, rec["losses"])
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
    if not args.no_eval and args.dataset == "gsm8k":
        # Free the trainer (optimizer state) before generating.
        t._trainer = None
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()
        nvml_mark("eval")
        ev = gsm_eval(t._model, t._tokenizer)
        rec["items_file"] = write_items(tag, ev)
        ev.pop("items")
        rec.update({k: v for k, v in ev.items() if k != "answer_loss"})
        rec["acc_wilson95"] = wilson(ev["acc_strict"], ev["n"])
        rec["heldout_after"] = ev["answer_loss"]
    elif not args.no_eval:
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
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    rec["s_per_step"] = round(statistics.median(gaps), 4) if gaps else None
    rec["steps_completed"] = len(losses)
    rec["s_per_step_measured_on"] = SPEED_LABEL
finally:
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    rec.update(nvml_report())
    shutil.rmtree(os.path.join(WORKDIR, tag), ignore_errors=True)

dump(os.path.join(RUNS, f"{tag}.json"), rec)
# exit 0 for ok and for an OOM under a cap (a measured "does not fit"); 1 otherwise
sys.exit(0 if rec["status"] == "ok" or (rec["status"] == "oom" and args.vram_cap_gb) else 1)
