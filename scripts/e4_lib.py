"""E4 harness library: full fine-tuning vs QLoRA on a code task (pure, CPU-importable).

Everything the E4 pod run decides or measures that is *not* a GPU call lives here,
so it can be unit-tested on a laptop (``tests/test_e4_harness.py``):

* dataset preparation for ``nvidia/OpenCodeInstruct`` (row filters, interface hint,
  normalised-hash dedupe, deterministic eval/train split, overlap counts);
* the pass@1 executor: model-generated code runs in a resource-limited subprocess,
  and **only** when ``allow_exec`` is set on a POSIX host (the pod). On Windows or
  without the flag it refuses;
* pass@1 aggregation, Wilson intervals, McNemar exact, paired bootstrap (the same
  statistics as stage d, ``scripts/pod_gsm8k_summary.py``, ported verbatim);
* the pre-registered decision rules as pure functions (``precheck_verdict``,
  ``premise_verdict``, ``ship_verdict``), so the verdict is computed, not eyeballed;
* the time-based budget guard and its explicit drop order.

The pre-registration these implement is
``docs/receipts/2026-10-e4-code/README.md``. If a number here disagrees with that
file, the file wins and this module is the bug.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess  # nosec B404 - the pass@1 sandbox spawns one interpreter per candidate
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------ constants
DATASET_ID = "nvidia/OpenCodeInstruct"
# Pinned: the Hub sha the licence evidence in the receipt README was read at.
DATASET_REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
DATASET_SHARD = "data/train-00000-of-00050.parquet"
DEFAULT_N_EVAL = 500
SPLIT_SEED = 0

# Pre-registered gates (docs/receipts/2026-10-e4-code/README.md).
PRECHECK_LOW, PRECHECK_HIGH = 0.10, 0.80  # strictly inside
LOSS_MARGIN_NATS = 0.05
ALPHA = 0.05
MIN_SEEDS = 2  # a gate needs at least this many completed seeds per arm

ASSISTANT_MARKER = "<|im_start|>assistant\n"

# Stage-d measured seconds per training step on the RTX 5090 (lead's brief).
S_PER_STEP = {
    ("3b", "default"): 0.317, ("3b", "qlora"): 0.434, ("3b", "block_k5"): 0.144,
    ("7b", "block_k5"): 0.220, ("7b", "qlora"): 0.534, ("7b", "galore"): 1.271,
}


# ------------------------------------------------------------ text utilities
def slug(model: str) -> str:
    """File-name form of a model id ("Qwen/Qwen2.5-7B" -> "Qwen_Qwen2.5-7B"); as in the driver."""
    if os.path.isdir(model):
        model = os.path.basename(os.path.normpath(model))
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in model)


def normalize_text(s: str) -> str:
    """Whitespace-collapsed, lower-cased text: the dedupe key's input."""
    return re.sub(r"\s+", " ", s).strip().lower()


def content_hash(s: str) -> str:
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()[:32]


def code_blocks(text: str) -> list[tuple[str, str]]:
    """Fenced blocks as (language, body). An unterminated last fence (a generation
    cut off at the token limit) is returned as a block too."""
    return [(lang.lower(), body) for lang, body in
            re.findall(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", text, re.S)]


def _ast_parse(src: str) -> ast.Module:
    """``ast.parse`` without the SyntaxWarnings that model-written / dataset strings trigger."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(src)


def _parses(code: str) -> bool:
    try:
        _ast_parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def extract_code(text: str) -> str | None:
    """The Python solution inside a model reply, or None.

    First python (or untagged) fenced block that defines something; failing that
    the first python block; with no fence at all, the whole reply if it parses.
    """
    py = [b for lang, b in code_blocks(text) if lang in ("python", "py", "python3", "")]
    if py:
        for b in py:
            if re.search(r"^\s*(async\s+def|def|class)\s", b, re.M):
                return b
        return py[0]
    return text if text.strip() and _parses(text) else None


# --------------------------------------------------------- dataset preparation
def _json_list(value: Any) -> list | None:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):  # numpy / arrow arrays
        return list(value.tolist())
    if isinstance(value, str):
        try:
            out = json.loads(value)
        except json.JSONDecodeError:
            return None
        return out if isinstance(out, list) else None
    return None


def interface_hint(code: str, tests: list[str]) -> str | None:
    """Signatures of the top-level names in ``code`` that the tests refer to.

    The OpenCodeInstruct questions name their function in only about half of the
    cases, while the tests call it by name; without a hint no model could pass.
    The hint is appended to every prompt (train and eval alike) so the format is
    learned, not a train/eval mismatch. Returns None when the tests refer to
    nothing the reference defines (such a row is dropped).
    """
    try:
        tree = _ast_parse(code)
        used: set[str] = set()
        for t in tests:
            for node in ast.walk(_ast_parse(t.strip())):
                if isinstance(node, ast.Name):
                    used.add(node.id)
    except (SyntaxError, ValueError):
        return None
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in used:
            lines.append(_signature(node))
        elif isinstance(node, ast.ClassDef) and node.name in used:
            lines.append(f"class {node.name}:")
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and (m.name == "__init__" or not m.name.startswith("_"))][:12]
            for m in methods:
                for dec in m.decorator_list:
                    if isinstance(dec, ast.Name):
                        lines.append(f"    @{dec.id}")
                lines.append("    " + _signature(m))
            if not methods:
                lines.append("    ...")
    return "\n".join(lines) if lines else None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{kw} {node.name}({ast.unparse(node.args)}){ret}: ..."


def build_user_prompt(problem: str, hint: str) -> str:
    return (f"{problem.strip()}\n\nYour solution must define this interface:\n"
            f"```python\n{hint}\n```\n"
            "Reply with the complete implementation in a single Python code block.")


def prepare_row(raw: dict) -> tuple[dict | None, str | None]:
    """One raw OpenCodeInstruct row -> (record, None) or (None, reject reason).

    Kept: ``domain == "generic"`` (the TACO-seeded "algorithmic" rows are left
    out: their seed licence is mixed), reference solution passed every generated
    unit test (``average_test_score == 1.0`` and every status "pass"), the
    reference parses and defines something the tests call.
    """
    if raw.get("domain") != "generic":
        return None, "not_generic"
    try:
        score = float(raw.get("average_test_score"))
    except (TypeError, ValueError):
        return None, "score_below_1"
    tests = _json_list(raw.get("unit_tests"))
    status = _json_list(raw.get("tests_execution_status"))
    if score != 1.0:
        return None, "score_below_1"
    if not tests or not all(isinstance(t, str) for t in tests):
        return None, "tests_empty"
    if not status or any(s != "pass" for s in status):
        return None, "tests_status_fail"
    output = raw.get("output") or ""
    blocks = [b for lang, b in code_blocks(output) if lang in ("python", "py", "python3", "")]
    if not blocks:
        return None, "no_code"
    code = blocks[0]
    if not _parses(code):
        return None, "syntax_error"
    if any(not _parses(t.strip()) for t in tests):
        return None, "bad_tests"
    hint = interface_hint(code, tests)
    if hint is None:
        return None, "no_interface"
    problem = (raw.get("input") or "").strip()
    if not problem:
        return None, "no_problem"
    return {
        "id": str(raw.get("id")), "problem": problem, "user": build_user_prompt(problem, hint),
        "solution_text": output.strip(), "code": code, "tests": tests, "hint": hint,
        "generation_algorithm": raw.get("generation_algorithm"),
        "p_hash": content_hash(problem), "s_hash": content_hash(code),
    }, None


def split_pool(records: list[dict], *, seed: int = SPLIT_SEED, n_eval: int = DEFAULT_N_EVAL
               ) -> tuple[list[dict], list[dict], dict]:
    """Deterministic eval / train split with overlap removal; returns (eval, train, report).

    Shuffle with ``random.Random(seed)``. Eval = the first ``n_eval`` records that
    are unique (by normalised prompt hash and by normalised solution hash) among
    the eval picks. Train = everything else, minus any record whose prompt hash
    or solution hash appears in eval (the overlap counts), minus repeats within
    train (first occurrence kept). Train order stays the shuffled order, so every
    arm that uses the same seed sees the same data order.
    """
    pool = list(records)
    random.Random(seed).shuffle(pool)  # nosec B311 - dataset split, not cryptography
    eval_rows: list[dict] = []
    eval_p: set[str] = set()
    eval_s: set[str] = set()
    eval_dup_skipped = 0
    rest: list[dict] = []
    for r in pool:
        if len(eval_rows) < n_eval:
            if r["p_hash"] in eval_p or r["s_hash"] in eval_s:
                eval_dup_skipped += 1
                rest.append(r)  # still a train candidate; the overlap rule below removes it
                continue
            eval_rows.append(r)
            eval_p.add(r["p_hash"])
            eval_s.add(r["s_hash"])
        else:
            rest.append(r)
    overlap_p = overlap_s = overlap_total = internal = 0
    train: list[dict] = []
    seen_p: set[str] = set()
    seen_s: set[str] = set()
    for r in rest:
        hit_p, hit_s = r["p_hash"] in eval_p, r["s_hash"] in eval_s
        if hit_p or hit_s:
            overlap_p += int(hit_p)
            overlap_s += int(hit_s)
            overlap_total += 1
            continue
        if r["p_hash"] in seen_p or r["s_hash"] in seen_s:
            internal += 1
            continue
        seen_p.add(r["p_hash"])
        seen_s.add(r["s_hash"])
        train.append(r)
    report = {
        "seed": seed, "pool": len(records), "n_eval": len(eval_rows), "n_train": len(train),
        "eval_internal_duplicates_skipped": eval_dup_skipped,
        "train_removed_overlap_with_eval": overlap_total,
        "train_removed_overlap_by_prompt_hash": overlap_p,
        "train_removed_overlap_by_solution_hash": overlap_s,
        "train_removed_internal_duplicates": internal,
        "dedupe_key": "sha256 of lower-cased whitespace-collapsed text; prompt = the question, "
                      "solution = the reference code block",
    }
    return eval_rows, train, report


def build_dataset(raw_rows: list[dict], *, seed: int = SPLIT_SEED, n_eval: int = DEFAULT_N_EVAL
                  ) -> tuple[list[dict], list[dict], dict]:
    """Raw rows -> (eval records, train records, report with every drop counted)."""
    kept: list[dict] = []
    rejects: dict[str, int] = {}
    for raw in raw_rows:
        rec, why = prepare_row(raw)
        if rec is None:
            rejects[why or "unknown"] = rejects.get(why or "unknown", 0) + 1
        else:
            kept.append(rec)
    eval_rows, train_rows, rep = split_pool(kept, seed=seed, n_eval=n_eval)
    rep.update({"raw_rows": len(raw_rows), "rejected": dict(sorted(rejects.items())),
                "valid_after_filters": len(kept)})
    return eval_rows, train_rows, rep


def to_messages(rec: dict) -> dict:
    """The ChatML-ready training / eval row (DatasetLoader reads ``messages``)."""
    return {"messages": [{"role": "user", "content": rec["user"]},
                         {"role": "assistant", "content": rec["solution_text"]}]}


def eval_meta(rec: dict, qid: int) -> dict:
    return {"qid": qid, "id": rec["id"], "tests": rec["tests"], "hint": rec["hint"],
            "generation_algorithm": rec["generation_algorithm"]}


def chatml_text(rec: dict) -> str:
    """The exact text DatasetLoader builds from ``to_messages`` (checked in tests)."""
    return (f"<|im_start|>user\n{rec['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['solution_text']}<|im_end|>")


def synthetic_rows(n: int, seed: int = 0) -> list[dict]:
    """Schema-faithful fake OpenCodeInstruct rows for the pod driver's CPU dry run."""
    rng = random.Random(seed)  # nosec B311 - fake data
    rows = []
    for i in range(n):
        k = rng.randint(1, 9)
        name = f"add_{k}_{i}"
        code = f"def {name}(x):\n    return x + {k}\n"
        tests = [f"\nassert {name}({a}) == {a + k}\n" for a in (0, 1, 5)]
        rows.append({
            "id": f"synthetic-{i}", "domain": "generic", "generation_algorithm": "self-instruct",
            "input": f"Write a function that adds {k} to its argument (variant {i}).",
            "output": f"```python\n{code}```", "unit_tests": json.dumps(tests),
            "tests_execution_status": json.dumps(["pass"] * len(tests)), "average_test_score": "1.0"})
    return rows


# ------------------------------------------------------------------- executor
class ExecRefused(RuntimeError):
    """The sandbox refuses to run model-generated code here."""


def exec_supported() -> bool:
    if os.name == "nt" or not sys.platform.startswith(("linux", "darwin")):
        return False
    try:
        import resource  # noqa: F401
    except ImportError:
        return False
    return True


def ensure_exec_allowed(allow_exec: bool) -> None:
    """Raise unless the caller passed ``allow_exec`` on a POSIX host with ``resource``."""
    if not allow_exec:
        raise ExecRefused("refusing to run generated code: pass --allow-exec (pod only)")
    if not exec_supported():
        raise ExecRefused(f"refusing to run generated code on {sys.platform}: "
                          "the sandbox needs POSIX resource limits (run it on the pod)")


# Runs inside the child interpreter. Limits first, then the candidate, then each
# test in the candidate's namespace; the result goes to a file (stdout is
# discarded, so a candidate that prints cannot forge it).
_RUNNER = r'''
import json, os, resource, socket, sys
spec = json.load(open(sys.argv[1]))
lim = spec["limits"]
def _cap(which, soft):
    try:
        resource.setrlimit(which, (soft, soft))
    except (ValueError, OSError):
        pass
_cap(resource.RLIMIT_CPU, lim["cpu_s"])
_cap(resource.RLIMIT_AS, lim["mem_bytes"])
_cap(resource.RLIMIT_FSIZE, lim["fsize_bytes"])
_cap(resource.RLIMIT_NOFILE, 64)
_cap(resource.RLIMIT_CORE, 0)
def _no_net(*a, **k):
    raise OSError("network disabled in the pass@1 sandbox")
socket.socket = _no_net
socket.create_connection = _no_net
socket.getaddrinfo = _no_net
res = {"load": "ok", "tests": []}
ns = {"__name__": "candidate"}
try:
    exec(compile(spec["code"], "<candidate>", "exec"), ns)
except BaseException as exc:
    res["load"] = (type(exc).__name__ + ": " + str(exc))[:300]
else:
    for t in spec["tests"]:
        try:
            exec(compile(t.strip(), "<test>", "exec"), ns)
            res["tests"].append(True)
        except BaseException as exc:
            res["tests"].append(False)
            res.setdefault("first_failure", (type(exc).__name__ + ": " + str(exc))[:300])
with open("result.json", "w") as fh:
    json.dump(res, fh)
'''


def run_candidate(code: str | None, tests: list[str], *, allow_exec: bool = False,
                  timeout_s: float = 10.0, mem_mb: int = 2048) -> dict:
    """Run ``code`` against ``tests`` in a fresh, resource-limited interpreter.

    Returns ``{"outcome", "tests_passed", "tests_total", "detail", "wall_s"}`` with
    outcome in pass | fail | load_error | timeout | crash | no_code.
    Raises ``ExecRefused`` unless ``allow_exec`` is true on a POSIX host.

    Isolation is process-level (own session, temp cwd, rlimits on CPU / address
    space / file size / open files, no core dumps, stdin/stdout/stderr discarded,
    sockets disabled in the interpreter, a minimal environment) and is meant for
    a throwaway pod container, not as a security boundary against a hostile
    program.
    """
    ensure_exec_allowed(allow_exec)
    total = len(tests)
    if code is None or not code.strip():
        return {"outcome": "no_code", "tests_passed": 0, "tests_total": total, "detail": None,
                "wall_s": 0.0}
    import signal

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="e4_exec_") as tmp:
        spec = {"code": code, "tests": tests,
                "limits": {"cpu_s": int(math.ceil(timeout_s)) + 1, "mem_bytes": mem_mb * 2**20,
                           "fsize_bytes": 10 * 2**20}}
        with open(os.path.join(tmp, "spec.json"), "w", encoding="utf-8") as fh:
            json.dump(spec, fh)
        with open(os.path.join(tmp, "runner.py"), "w", encoding="utf-8") as fh:
            fh.write(_RUNNER)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        # nosec B603: argv is [this interpreter, "-I", a runner file we just wrote, spec file];
        # no shell, no user-controlled argv. The candidate code is data in spec.json.
        proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-I", "runner.py", "spec.json"], cwd=tmp, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            with contextlib.suppress(ProcessLookupError, PermissionError):  # already gone: fine
                os.killpg(proc.pid, signal.SIGKILL)  # the whole session, children included
            proc.wait()
        wall = round(time.perf_counter() - t0, 3)
        result_path = os.path.join(tmp, "result.json")
        res = None
        if os.path.exists(result_path):
            try:
                with open(result_path, encoding="utf-8") as fh:
                    res = json.load(fh)
            except (OSError, ValueError):
                res = None
    if timed_out:
        return {"outcome": "timeout", "tests_passed": 0, "tests_total": total,
                "detail": f"wall clock > {timeout_s}s", "wall_s": wall}
    if res is None:
        rc = proc.returncode
        outcome = "timeout" if rc == -signal.SIGXCPU else "crash"
        return {"outcome": outcome, "tests_passed": 0, "tests_total": total,
                "detail": f"exit status {rc}", "wall_s": wall}
    if res.get("load") != "ok":
        return {"outcome": "load_error", "tests_passed": 0, "tests_total": total,
                "detail": res.get("load"), "wall_s": wall}
    passed = sum(1 for x in res.get("tests", []) if x is True)
    ok = total > 0 and passed == total
    return {"outcome": "pass" if ok else "fail", "tests_passed": passed, "tests_total": total,
            "detail": res.get("first_failure"), "wall_s": wall}


def run_many(jobs: list[tuple[str | None, list[str]]], *, allow_exec: bool, workers: int = 8,
             timeout_s: float = 10.0, mem_mb: int = 2048) -> list[dict]:
    """``run_candidate`` over many (code, tests) jobs, in order, on a thread pool."""
    ensure_exec_allowed(allow_exec)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(run_candidate, c, t, allow_exec=allow_exec, timeout_s=timeout_s,
                            mem_mb=mem_mb) for c, t in jobs]
        return [f.result() for f in futs]


def not_executed(tests: list[str]) -> dict:
    return {"outcome": "not_executed", "tests_passed": 0, "tests_total": len(tests),
            "detail": "dry run: sandbox skipped", "wall_s": 0.0}


def summarize_outcomes(outcomes: list[dict]) -> dict:
    """Counts per outcome and pass@1 (greedy decoding: one sample per problem)."""
    n = len(outcomes)
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
    passed = counts.get("pass", 0)
    executed = n - counts.get("not_executed", 0)
    return {"n": n, "pass_at_1": round(passed / n, 4) if n else None, "passed": passed,
            "executed": executed > 0, "outcomes": dict(sorted(counts.items()))}


# ----------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def spread(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 4),
            "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 4), "max": round(max(xs), 4)}


def mcnemar_exact(a, b) -> dict:
    """Exact two-sided McNemar on paired boolean arrays (a_only vs b_only)."""
    import numpy as np

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)
    return {"a_only": b01, "b_only": b10, "p": round(p, 5)}


def paired_bootstrap(acc_a, acc_b, loss_a, loss_b, tok_a, tok_b, n_boot: int = 10000) -> dict:
    """Paired bootstrap (items resampled, seed 0) of the pass@1 and held-out-loss differences."""
    import numpy as np

    acc_a, acc_b = np.asarray(acc_a, dtype=float), np.asarray(acc_b, dtype=float)
    loss_a, loss_b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    tok_a, tok_b = np.asarray(tok_a, dtype=float), np.asarray(tok_b, dtype=float)
    rng = np.random.default_rng(0)
    m = len(acc_a)
    idx = rng.integers(0, m, size=(n_boot, m))
    dacc = acc_a[idx].mean(1) - acc_b[idx].mean(1)
    dloss = loss_a[idx].sum(1) / tok_a[idx].sum(1) - loss_b[idx].sum(1) / tok_b[idx].sum(1)
    return {
        "acc_diff": round(float(acc_a.mean() - acc_b.mean()), 4),
        "acc_diff_ci95": [round(float(np.quantile(dacc, 0.025)), 4),
                          round(float(np.quantile(dacc, 0.975)), 4)],
        "loss_diff": round(float(loss_a.sum() / tok_a.sum() - loss_b.sum() / tok_b.sum()), 4),
        "loss_diff_ci95": [round(float(np.quantile(dloss, 0.025)), 4),
                           round(float(np.quantile(dloss, 0.975)), 4)],
    }


def item_arrays(items: list[dict]):
    """Per-item records -> (correct bool[], loss_sum[], tokens[], qids), sorted by qid."""
    import numpy as np

    its = sorted(items, key=lambda x: x["qid"])
    return (np.array([bool(x["correct"]) for x in its]),
            np.array([x["answer_loss_sum"] for x in its], dtype=float),
            np.array([x["answer_tokens"] for x in its], dtype=float),
            [x["qid"] for x in its])


def majority_correct(per_seed_correct: list):
    """Item solved by strictly more than half of an arm's seeds (1 seed: that seed)."""
    import numpy as np

    return np.mean(np.stack([np.asarray(c, dtype=float) for c in per_seed_correct]), axis=0) > 0.5


def pair_stats(a_by_seed: dict[int, list[dict]], b_by_seed: dict[int, list[dict]],
               n_boot: int = 10000) -> dict:
    """Paired statistics for arm A against arm B from per-seed item records.

    * ``pooled_over_seeds`` (stage d's definition): per item, each arm's mean over its
      seeds, then the paired bootstrap. ``loss_diff`` is A - B in nats (negative = A better).
    * ``majority_vote``: an item counts as solved by an arm if solved in strictly more
      than half of its seeds; McNemar exact on those binary vectors. This is the
      accuracy test the decision rules use.
    * ``per_seed``: seed-to-seed pairs (B's first seed when B lacks the seed).
    """
    import numpy as np

    pa = {s: item_arrays(v) for s, v in a_by_seed.items()}
    pb = {s: item_arrays(v) for s, v in b_by_seed.items()}
    qids = None
    for arr in list(pa.values()) + list(pb.values()):
        if qids is None:
            qids = arr[3]
        elif arr[3] != qids:
            raise ValueError("arms were not evaluated on the same qids")
    if not pa or not pb:
        raise ValueError("both arms need at least one seed")
    per_seed = []
    first_b = pb[sorted(pb)[0]]
    for s in sorted(pa):
        rb_seed = s if s in pb else sorted(pb)[0]
        b = pb.get(s, first_b)
        a = pa[s]
        per_seed.append({"seed_a": s, "seed_b": rb_seed, "mcnemar": mcnemar_exact(a[0], b[0]),
                         **paired_bootstrap(a[0], b[0], a[1], b[1], a[2], b[2], n_boot)})
    mean_a = [np.mean([x[i] for x in pa.values()], axis=0) for i in range(3)]
    mean_b = [np.mean([x[i] for x in pb.values()], axis=0) for i in range(3)]
    pooled = paired_bootstrap(mean_a[0], mean_b[0], mean_a[1], mean_b[1], mean_a[2], mean_b[2], n_boot)
    maj_a = majority_correct([x[0] for x in pa.values()])
    maj_b = majority_correct([x[0] for x in pb.values()])
    return {"seeds_a": sorted(pa), "seeds_b": sorted(pb), "n_items": len(qids or []),
            "pooled_over_seeds": pooled,
            "majority_vote": {"acc_a": round(float(maj_a.mean()), 4), "acc_b": round(float(maj_b.mean()), 4),
                              "mcnemar": mcnemar_exact(maj_a, maj_b)},
            "per_seed": per_seed}


# ------------------------------------------------------------- decision rules
def precheck_verdict(pass_at_1: float | None, heldout_loss: float | None) -> dict:
    """Abort gate on the untrained base model: 0.10 < pass@1 < 0.80, loss recorded."""
    reasons = []
    if pass_at_1 is None or not isinstance(pass_at_1, (int, float)) or math.isnan(pass_at_1):
        reasons.append("pass@1 not measured")
    elif not PRECHECK_LOW < pass_at_1 < PRECHECK_HIGH:
        reasons.append(f"pass@1 {pass_at_1:.4f} is not strictly inside ({PRECHECK_LOW}, {PRECHECK_HIGH})")
    if heldout_loss is None or not math.isfinite(heldout_loss):
        reasons.append("held-out loss not recorded")
    return {"ok": not reasons, "verdict": "PROCEED" if not reasons else "ABORT", "reasons": reasons,
            "rule": f"abort unless {PRECHECK_LOW} < base pass@1 < {PRECHECK_HIGH} "
                    "and the base held-out loss is recorded"}


def beats(pair: dict, *, loss_margin: float = LOSS_MARGIN_NATS, alpha: float = ALPHA) -> dict:
    """Does arm A beat arm B? The stage-d rule.

    A beats B if the pooled held-out-loss difference (A - B) is at least
    ``loss_margin`` nats lower, OR A's majority-vote pass@1 is higher with a
    McNemar exact p below ``alpha``.
    """
    loss_diff = pair["pooled_over_seeds"]["loss_diff"]
    mc = pair["majority_vote"]["mcnemar"]
    loss_ok = loss_diff <= -loss_margin
    acc_ok = mc["a_only"] > mc["b_only"] and mc["p"] < alpha
    lo, hi = pair["pooled_over_seeds"]["loss_diff_ci95"]
    return {"beats": bool(loss_ok or acc_ok), "loss_criterion": bool(loss_ok),
            "accuracy_criterion": bool(acc_ok), "loss_diff": loss_diff, "loss_diff_ci95": [lo, hi],
            "loss_ci_excludes_zero": bool(hi < 0.0 or lo > 0.0),
            "mcnemar": mc, "rule": f"loss_diff <= -{loss_margin} OR (a_only > b_only AND McNemar p < {alpha})"}


def _gate(pair: dict | None, seeds_a: int, seeds_b: int, *, name: str, min_seeds: int) -> dict:
    if pair is None or seeds_a < min_seeds or seeds_b < min_seeds:
        return {"gate": name, "verdict": "INCONCLUSIVE", "proceed": False,
                "reason": f"needs >= {min_seeds} completed seeds per arm (have {seeds_a} and {seeds_b})"}
    b = beats(pair)
    return {"gate": name, "verdict": "PASS" if b["beats"] else "FAIL", "proceed": b["beats"], **b}


def premise_verdict(pair_full_vs_qlora: dict | None, seeds_full: int, seeds_qlora: int,
                    *, min_seeds: int = MIN_SEEDS) -> dict:
    """3B premise gate: standard full FT (A) vs QLoRA r=256 (B). FAIL or INCONCLUSIVE -> stop."""
    return _gate(pair_full_vs_qlora, seeds_full, seeds_qlora, name="premise_3b_full_ft_beats_qlora",
                 min_seeds=min_seeds)


def ship_verdict(pair_engine_b_vs_qlora: dict | None, seeds_engine_b: int, seeds_qlora: int,
                 *, min_seeds: int = MIN_SEEDS) -> dict:
    """7B ship rule: engine B (A) vs QLoRA r=256 (B). PASS -> may leave experimental."""
    return _gate(pair_engine_b_vs_qlora, seeds_engine_b, seeds_qlora,
                 name="ship_7b_engine_b_beats_qlora", min_seeds=min_seeds)


# --------------------------------------------------------------------- budget
@dataclass(frozen=True)
class RunSpec:
    size: str
    arm: str
    seed: int

    @property
    def tag(self) -> str:
        return f"e4_{self.size}_{self.arm}_s{self.seed}"


def plan_for(size: str, seeds: tuple[int, ...] | None = None) -> list[RunSpec]:
    """The run order for a stage. The budget guard drops from the END of this list,
    and never runs a later entry after dropping an earlier one.

    3B: the premise gate's arms first (full FT and QLoRA, interleaved by seed so
    the gate has two complete seeds as early as possible), engine B last.
    Drop order: engine B s2, s1, s0, then QLoRA s2, full FT s2, ...
    7B: engine B and QLoRA interleaved (the ship rule), GaLore last: it is the most
    expensive arm and adopting it is the Director's call. Drop order: GaLore, QLoRA s1, engine B s1, ...
    """
    if size == "3b":
        s = seeds if seeds is not None else (0, 1, 2)
        return ([RunSpec("3b", a, x) for x in s for a in ("default", "qlora")]
                + [RunSpec("3b", "block_k5", x) for x in s])
    if size == "7b":
        s = seeds if seeds is not None else (0, 1)
        return ([RunSpec("7b", a, x) for x in s for a in ("block_k5", "qlora")]
                + [RunSpec("7b", "galore", 0)])
    raise ValueError(f"unknown size {size!r}")


def estimate_run_s(spec: RunSpec, steps: int, *, eval_s: float, load_s: float = 60.0,
                   overhead_s: float = 30.0, measured_wall_s: float | None = None,
                   margin: float = 1.10) -> float:
    """Wall-clock estimate for one run. A completed run of the same arm (``measured_wall_s``,
    scaled to ``steps`` by the caller) replaces the model; otherwise stage d's s/step."""
    if measured_wall_s is not None:
        return measured_wall_s * margin
    return (steps * S_PER_STEP[(spec.size, spec.arm)] + eval_s + load_s + overhead_s) * margin


@dataclass
class BudgetGuard:
    """Time-based spend cap. ``rate_usd_h`` is the pod's hourly price; the deadline is
    ``start + cap_usd / rate`` minus ``reserve_s`` kept for summaries and receipt copy.

    ``admit`` is called with each run's estimated duration in plan order. The first
    run that would end past the deadline is dropped and so is every later run
    (sticky): priority order is never inverted by a cheap run squeezing in.
    """

    cap_usd: float
    rate_usd_h: float = 0.90
    start_epoch: float = field(default_factory=time.time)
    reserve_s: float = 600.0
    dropped: list[dict] = field(default_factory=list)
    _cut: bool = False

    @property
    def deadline_epoch(self) -> float:
        return self.start_epoch + self.cap_usd / self.rate_usd_h * 3600.0 - self.reserve_s

    def admit(self, spec: RunSpec, est_s: float, now: float | None = None) -> tuple[bool, str]:
        now = time.time() if now is None else now
        if self._cut:
            reason = "an earlier, higher-priority run was dropped (sticky)"
        elif now + est_s > self.deadline_epoch:
            self._cut = True
            reason = (f"would end {now + est_s - self.deadline_epoch:.0f}s past the deadline "
                      f"(cap ${self.cap_usd:.2f} at ${self.rate_usd_h:.2f}/h, reserve {self.reserve_s:.0f}s)")
        else:
            return True, "fits"
        self.dropped.append({"tag": spec.tag, "est_s": round(est_s), "reason": reason})
        return False, reason


def walk_plan(plan: list[RunSpec], guard: BudgetGuard, est: Callable[[RunSpec], float],
              now: float) -> list[RunSpec]:
    """Which plan entries a guard would admit, running back to back from ``now``
    (used for the README estimate and for tests; the live driver calls ``admit`` itself)."""
    admitted = []
    t = now
    for spec in plan:
        e = est(spec)
        ok, _ = guard.admit(spec, e, t)
        if ok:
            admitted.append(spec)
            t += e
    return admitted


# -------------------------------------------------------------------- selftest
_SELFTEST = [  # fixed hand-written snippets (never model output): (name, code, expected outcome)
    ("pass", "def add(a, b):\n    return a + b\n", "pass"),
    ("fail", "def add(a, b):\n    return a - b\n", "fail"),
    ("raises", "raise ValueError('boom')\n", "load_error"),
    ("infinite loop", "while True:\n    pass\n", "timeout"),
    ("hard exit", "import os\nos._exit(3)\n", "crash"),
]


def selftest(allow_exec: bool) -> int:
    """Run the fixed snippets through the sandbox; 0 only if every outcome is as expected.

    ``python scripts/e4_lib.py selftest --allow-exec`` on the pod before the paid stages."""
    tests = ["assert add(1, 2) == 3"]
    bad = 0
    for name, code, want in _SELFTEST:
        got = run_candidate(code, tests, allow_exec=allow_exec, timeout_s=2.0)["outcome"]
        print(f"selftest {name}: {got} (expected {want})", flush=True)
        bad += int(got != want)
    print("SELFTEST " + ("OK" if not bad else f"FAILED ({bad})"), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        try:
            sys.exit(selftest("--allow-exec" in sys.argv))
        except ExecRefused as exc:
            print(f"SELFTEST REFUSED: {exc}", flush=True)
            sys.exit(2)
    print("usage: python scripts/e4_lib.py selftest [--allow-exec]", file=sys.stderr)
    sys.exit(2)
