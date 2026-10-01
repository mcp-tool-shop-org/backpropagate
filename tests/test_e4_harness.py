"""CPU tests for the E4 code-task harness (scripts/e4_lib.py and the pod scripts around it).

Experiment E4 asks where full fine-tuning beats QLoRA on a code task. Its decision rules are
pre-registered in docs/receipts/2026-10-e4-code/README.md; these tests pin the pure parts that
compute them: dataset preparation, the pass@1 sandbox (fixed hand-written snippets only, never
model output), the paired statistics, the verdict functions and the budget guard.

The sandbox tests need POSIX ``resource`` and are skipped on Windows; the refusal guard is
tested everywhere.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import e4_lib as E4  # noqa: E402
import pod_e4  # noqa: E402
import pod_e4_summary  # noqa: E402

posix_only = pytest.mark.skipif(not E4.exec_supported(), reason="the sandbox needs POSIX resource limits")


# ------------------------------------------------------------------ helpers
def raw_row(i: int = 0, **over) -> dict:
    row = {
        "id": f"id-{i}", "domain": "generic", "generation_algorithm": "self-instruct",
        "input": f"Write a function that doubles its argument (variant {i}).",
        "output": f"```python\ndef double_{i}(x):\n    return 2 * x\n```",
        "unit_tests": json.dumps([f"\nassert double_{i}(2) == 4\n", f"\nassert double_{i}(0) == 0\n"]),
        "tests_execution_status": json.dumps(["pass", "pass"]), "average_test_score": "1.0",
    }
    row.update(over)
    return row


def records(n: int) -> list[dict]:
    out = []
    for i in range(n):
        rec, why = E4.prepare_row(raw_row(i))
        assert rec is not None, why
        out.append(rec)
    return out


# ------------------------------------------------------------ dataset prep
class TestPrepareRow:
    def test_accepts_a_good_row(self):
        rec, why = E4.prepare_row(raw_row(3))
        assert why is None and rec is not None
        assert rec["hint"] == "def double_3(x): ..."
        assert rec["user"].startswith("Write a function that doubles")
        assert "```python\ndef double_3(x): ...\n```" in rec["user"]
        assert rec["code"].startswith("def double_3")

    @pytest.mark.parametrize("over,reason", [
        ({"domain": "algorithmic"}, "not_generic"),
        ({"average_test_score": "0.9"}, "score_below_1"),
        ({"average_test_score": None}, "score_below_1"),
        ({"unit_tests": "[]"}, "tests_empty"),
        ({"tests_execution_status": json.dumps(["pass", "fail"])}, "tests_status_fail"),
        ({"output": "no code here"}, "no_code"),
        ({"output": "```python\ndef f(:\n```"}, "syntax_error"),
        ({"unit_tests": json.dumps(["assert (", "assert 1"])}, "bad_tests"),
        ({"unit_tests": json.dumps(["\nassert unrelated(1) == 1\n"])}, "no_interface"),
        ({"input": "   "}, "no_problem"),
    ])
    def test_rejects(self, over, reason):
        rec, why = E4.prepare_row(raw_row(0, **over))
        assert rec is None and why == reason

    def test_unit_tests_may_arrive_as_a_list(self):
        rec, why = E4.prepare_row(raw_row(0, unit_tests=["\nassert double_0(1) == 2\n"],
                                          tests_execution_status=["pass"]))
        assert why is None and rec["tests"] == ["\nassert double_0(1) == 2\n"]


class TestInterfaceHint:
    def test_function_with_defaults_and_annotations(self):
        code = "def f(a: int, b=3) -> int:\n    return a + b\n\ndef helper():\n    pass\n"
        assert E4.interface_hint(code, ["assert f(1) == 4"]) == "def f(a: int, b=3) -> int: ..."

    def test_class_lists_public_methods_and_init_only(self):
        code = ("class Bank:\n    def __init__(self, start=0):\n        pass\n"
                "    def deposit(self, amount):\n        pass\n    def _secret(self):\n        pass\n"
                "    @staticmethod\n    def rate():\n        return 1\n")
        hint = E4.interface_hint(code, ["b = Bank(); b.deposit(1)"])
        assert hint == ("class Bank:\n    def __init__(self, start=0): ...\n"
                        "    def deposit(self, amount): ...\n    @staticmethod\n    def rate(): ...")

    def test_none_when_tests_call_nothing_defined(self):
        assert E4.interface_hint("def f():\n    pass\n", ["assert g() == 1"]) is None

    def test_none_on_unparsable_input(self):
        assert E4.interface_hint("def f(:", ["assert f()"]) is None


class TestExtractCode:
    def test_first_python_block_that_defines_something(self):
        text = "Use:\n```python\nprint(1)\n```\nthen\n```python\ndef f():\n    return 1\n```"
        assert E4.extract_code(text) == "def f():\n    return 1\n"

    def test_falls_back_to_first_python_block(self):
        assert E4.extract_code("```python\nx = 1\n```") == "x = 1\n"

    def test_unterminated_fence_from_a_truncated_generation(self):
        assert E4.extract_code("```python\ndef f():\n    return 1") == "def f():\n    return 1"

    def test_untagged_fence(self):
        assert E4.extract_code("```\ndef f():\n    return 1\n```") == "def f():\n    return 1\n"

    def test_ignores_other_languages(self):
        assert E4.extract_code("```bash\nls\n```") is None

    def test_bare_code_without_fence(self):
        assert E4.extract_code("def f():\n    return 1\n") == "def f():\n    return 1\n"

    def test_prose_is_not_code(self):
        assert E4.extract_code("I think the answer is two.") is None


class TestSplit:
    def test_deterministic_for_a_fixed_seed_and_different_for_another(self):
        recs = records(60)
        e1, t1, r1 = E4.split_pool(recs, seed=0, n_eval=10)
        e2, t2, r2 = E4.split_pool(list(recs), seed=0, n_eval=10)
        assert [r["id"] for r in e1] == [r["id"] for r in e2]
        assert [r["id"] for r in t1] == [r["id"] for r in t2]
        assert r1 == r2
        e3, _, _ = E4.split_pool(recs, seed=1, n_eval=10)
        assert [r["id"] for r in e1] != [r["id"] for r in e3]

    def test_sizes_and_disjointness(self):
        recs = records(60)
        ev, tr, rep = E4.split_pool(recs, seed=0, n_eval=10)
        assert len(ev) == 10 and len(tr) == 50 == rep["n_train"]
        assert not {r["id"] for r in ev} & {r["id"] for r in tr}

    def test_overlap_by_solution_and_prompt_hash_is_removed_and_counted(self):
        recs = records(40)
        # clones: same solution as record 0 (new wording) and same prompt as record 1 (new solution)
        clone_s = dict(recs[0], id="clone-s", p_hash="p-unique-1", user="x")
        clone_p = dict(recs[1], id="clone-p", s_hash="s-unique-1", user="y")
        ev, tr, rep = E4.split_pool(recs + [clone_s, clone_p], seed=0, n_eval=15)
        ev_p, ev_s = {r["p_hash"] for r in ev}, {r["s_hash"] for r in ev}
        assert all(r["p_hash"] not in ev_p and r["s_hash"] not in ev_s for r in tr)
        # nothing is lost silently: every record is in eval, in train, or counted as removed
        assert len(ev) + len(tr) + rep["train_removed_overlap_with_eval"] \
            + rep["train_removed_internal_duplicates"] == 42

    def test_a_clone_of_an_eval_item_is_counted_as_overlap(self):
        recs = records(30)
        ev0, _, _ = E4.split_pool(recs, seed=0, n_eval=5)
        twin = dict(ev0[0], id="twin-of-eval")  # same hashes as an item that will be in eval
        _, tr, rep = E4.split_pool(recs + [twin], seed=0, n_eval=5)
        assert "twin-of-eval" not in {r["id"] for r in tr}
        assert rep["train_removed_overlap_with_eval"] >= 1
        assert rep["train_removed_overlap_by_solution_hash"] >= 1
        assert rep["train_removed_overlap_by_prompt_hash"] >= 1

    def test_train_internal_duplicates_keep_first(self):
        recs = records(30)
        dup = dict(recs[5], id="dup-of-5")
        ev, tr, rep = E4.split_pool(recs + [dup, dict(dup, id="dup-2")], seed=0, n_eval=3)
        twins = [r for r in tr if r["s_hash"] == recs[5]["s_hash"]]
        assert len(twins) <= 1
        assert rep["pool"] == 32

    def test_eval_picks_are_unique_among_themselves(self):
        recs = records(10)
        recs += [dict(r, id=r["id"] + "-b") for r in recs]  # every record twice
        ev, tr, rep = E4.split_pool(recs, seed=0, n_eval=10)
        assert len({r["s_hash"] for r in ev}) == len(ev) == 10
        assert tr == [] and rep["eval_internal_duplicates_skipped"] > 0

    def test_build_dataset_counts_every_reject(self):
        rows = [raw_row(i) for i in range(20)]
        rows += [raw_row(100, domain="algorithmic"), raw_row(101, average_test_score="0.5"),
                 raw_row(102, output="prose only")]
        ev, tr, rep = E4.build_dataset(rows, seed=0, n_eval=5)
        assert rep["raw_rows"] == 23 and rep["valid_after_filters"] == 20
        assert rep["rejected"] == {"no_code": 1, "not_generic": 1, "score_below_1": 1}
        assert len(ev) == 5 and len(tr) == 15


class TestFitFilter:
    """Rows that do not fit --seq are dropped (never truncated), before the eval split is drawn."""

    def long_rows(self, n_short=40, n_long=15):
        rows = [raw_row(i) for i in range(n_short)]
        for i in range(n_long):
            j = 1000 + i
            doc = "padding " * 200
            rows.append(raw_row(j, output=f'```python\ndef double_{j}(x):\n    """{doc}"""\n    return 2 * x\n```'))
        return rows

    @staticmethod
    def words(rec):  # a stand-in tokenizer: one token per whitespace-separated word
        return len(E4.chatml_text(rec).split())

    def test_boundary_and_reserve(self):
        recs = records(3)
        n = self.words(recs[0])
        kept, dropped = E4.fit_filter(list(recs), self.words, max_tokens=n + E4.FIT_RESERVE_TOKENS)
        assert len(kept) == 3 and dropped == 0
        kept, dropped = E4.fit_filter(list(recs), self.words, max_tokens=n + E4.FIT_RESERVE_TOKENS - 1)
        assert kept == [] and dropped == 3

    def test_filter_runs_before_the_eval_split_and_nothing_long_survives_in_either_pool(self):
        rows = self.long_rows()
        limit = 120
        ev, tr, rep = E4.build_dataset(rows, n_eval=10, length_fn=self.words, max_tokens=limit)
        assert rep["fit_filter"]["dropped_from_pool"] == 15
        assert rep["fit_filter"]["valid_before_filter"] == 55
        assert all(self.words(r) + E4.FIT_RESERVE_TOKENS <= limit for r in ev + tr)
        assert len(ev) == 10 and len(ev) + len(tr) + rep["train_removed_overlap_with_eval"] \
            + rep["train_removed_internal_duplicates"] == 40
        fit = rep["fit_filter"]
        assert fit["unfiltered_split_would_have_dropped_from_eval"] \
            + fit["unfiltered_split_would_have_dropped_from_train"] == 15

    def test_without_a_length_function_nothing_is_filtered(self):
        _, _, rep = E4.build_dataset(self.long_rows(), n_eval=10)
        assert rep["fit_filter"] is None

    def test_records_get_their_token_count(self):
        ev, tr, _ = E4.build_dataset(self.long_rows(), n_eval=5, length_fn=self.words, max_tokens=120)
        assert all(r["n_tokens"] == self.words(r) for r in ev + tr)

    def test_split_is_deterministic_after_filtering(self):
        a = E4.build_dataset(self.long_rows(), n_eval=10, length_fn=self.words, max_tokens=120)
        b = E4.build_dataset(self.long_rows(), n_eval=10, length_fn=self.words, max_tokens=120)
        assert [r["id"] for r in a[0]] == [r["id"] for r in b[0]]
        assert [r["id"] for r in a[1]] == [r["id"] for r in b[1]]


class TestShards:
    def loader(self, calls):
        shards = {"s0": [raw_row(i) for i in range(30)], "s1": [raw_row(100 + i) for i in range(30)]}

        def load(name):
            calls.append(name)
            return shards[name]

        return load

    def test_one_shard_is_enough_when_the_train_pool_is_large_enough(self):
        calls: list[str] = []
        ev, tr, rep = E4.build_with_shards(self.loader(calls), ("s0", "s1"), min_train=20, n_eval=5)
        assert calls == ["s0"] and rep["shards_used"] == ["s0"] and rep["enough_for_one_epoch"] is True
        assert len(tr) == 25 and len(ev) == 5

    def test_second_shard_is_added_when_train_is_short(self):
        calls: list[str] = []
        ev, tr, rep = E4.build_with_shards(self.loader(calls), ("s0", "s1"), min_train=40, n_eval=5)
        assert calls == ["s0", "s1"] and rep["shards_used"] == ["s0", "s1"]
        assert len(tr) == 55 and rep["enough_for_one_epoch"] is True

    def test_reports_when_even_every_shard_is_not_enough(self):
        calls: list[str] = []
        _, tr, rep = E4.build_with_shards(self.loader(calls), ("s0", "s1"), min_train=500, n_eval=5)
        assert rep["shards_used"] == ["s0", "s1"] and rep["enough_for_one_epoch"] is False
        assert len(tr) == 55

    def test_the_real_shards_are_pinned_to_one_revision(self):
        assert len(E4.DATASET_SHARDS) == 2 and E4.DATASET_SHARDS[0] == E4.DATASET_SHARD
        assert len(E4.DATASET_REVISION) == 40


class TestFormats:
    def test_messages_and_chatml_text_agree_with_the_library_loader(self, tmp_path):
        pytest.importorskip("datasets")
        from backpropagate.datasets import DatasetLoader

        rec = records(1)[0]
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps(E4.to_messages(rec)) + "\n", encoding="utf-8")
        text = DatasetLoader(str(path), validate=False).to_hf_dataset()["text"][0]
        assert text == E4.chatml_text(rec)
        assert text.endswith("<|im_end|>") and E4.ASSISTANT_MARKER in text

    def test_synthetic_rows_all_pass_the_filters(self):
        rows = E4.synthetic_rows(12)
        ev, tr, rep = E4.build_dataset(rows, n_eval=4)
        assert rep["rejected"] == {} and len(ev) == 4 and len(tr) == 8

    def test_eval_meta_carries_tests_and_hint(self):
        rec = records(1)[0]
        m = E4.eval_meta(rec, 7)
        assert m["qid"] == 7 and m["tests"] == rec["tests"] and m["hint"] == rec["hint"]

    def test_normalisation_collapses_case_and_whitespace(self):
        assert E4.content_hash("Def  f():\n  pass") == E4.content_hash("def f(): pass")
        assert E4.content_hash("a") != E4.content_hash("b")

    def test_slug(self):
        assert E4.slug("Qwen/Qwen2.5-7B") == "Qwen_Qwen2.5-7B"


# ---------------------------------------------------------------- executor
GOOD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]


class TestExecutorGuard:
    def test_refuses_without_allow_exec(self):
        with pytest.raises(E4.ExecRefused, match="--allow-exec"):
            E4.run_candidate(GOOD, TESTS)

    def test_refuses_many_without_allow_exec(self):
        with pytest.raises(E4.ExecRefused):
            E4.run_many([(GOOD, TESTS)], allow_exec=False)

    def test_refuses_when_the_host_has_no_posix_limits(self, monkeypatch):
        monkeypatch.setattr(E4, "exec_supported", lambda: False)
        with pytest.raises(E4.ExecRefused, match="POSIX"):
            E4.run_candidate(GOOD, TESTS, allow_exec=True)

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only")
    def test_refuses_on_windows_even_with_the_flag(self):
        assert E4.exec_supported() is False
        with pytest.raises(E4.ExecRefused):
            E4.run_candidate(GOOD, TESTS, allow_exec=True)

    def test_not_executed_record_is_not_a_pass(self):
        out = E4.not_executed(TESTS)
        assert out["outcome"] == "not_executed" and out["tests_total"] == 2
        s = E4.summarize_outcomes([out, out])
        assert s["executed"] is False and s["passed"] == 0


@posix_only
class TestExecutorSandbox:
    """Fixed hand-written snippets only."""

    def run(self, code, tests=TESTS, **kw):
        return E4.run_candidate(code, tests, allow_exec=True, **kw)

    def test_passing_snippet(self):
        r = self.run(GOOD)
        assert r["outcome"] == "pass" and r["tests_passed"] == r["tests_total"] == 2

    def test_failing_snippet(self):
        r = self.run("def add(a, b):\n    return a - b\n")
        assert r["outcome"] == "fail" and r["tests_passed"] == 1
        assert "AssertionError" in r["detail"]

    def test_infinite_loop_times_out_and_is_killed(self):
        r = self.run("while True:\n    pass\n", timeout_s=1.0)
        assert r["outcome"] == "timeout"

    def test_sleep_times_out_on_wall_clock(self):
        r = self.run("import time\ntime.sleep(30)\n", timeout_s=1.0)
        assert r["outcome"] == "timeout"

    def test_snippet_that_raises_at_load(self):
        r = self.run("raise ValueError('boom')\n")
        assert r["outcome"] == "load_error" and "ValueError" in r["detail"]

    def test_raising_inside_a_test_is_a_failure_not_a_crash(self):
        r = self.run("def add(a, b):\n    raise RuntimeError('nope')\n")
        assert r["outcome"] == "fail" and r["tests_passed"] == 0

    def test_syntax_error_is_a_load_error(self):
        assert self.run("def add(:\n")["outcome"] == "load_error"

    def test_sys_exit_does_not_pass_or_escape(self):
        assert self.run("import sys\nsys.exit(0)\n")["outcome"] == "load_error"

    def test_hard_exit_is_a_crash(self):
        r = self.run("import os\nos._exit(3)\n")
        assert r["outcome"] == "crash" and "3" in r["detail"]

    def test_memory_limit_bites(self):
        r = self.run("x = bytearray(1024 * 1024 * 1024)\n", mem_mb=256)
        assert r["outcome"] in ("load_error", "crash")

    def test_network_is_disabled_inside_the_candidate(self):
        r = self.run("import socket\nsocket.socket()\n")
        assert r["outcome"] == "load_error" and "network" in r["detail"]

    def test_candidate_files_land_in_a_temp_dir_not_the_cwd(self):
        name = "e4_sandbox_probe.txt"
        r = self.run(f"open({name!r}, 'w').write('x')\n" + GOOD)
        assert r["outcome"] == "pass"
        assert not os.path.exists(name)

    def test_printing_cannot_forge_the_result(self):
        code = "print('{\"load\": \"ok\", \"tests\": [true, true]}')\ndef add(a, b):\n    return 0\n"
        assert self.run(code)["outcome"] == "fail"

    def test_main_guard_does_not_fire(self):
        code = "def add(a, b):\n    return a + b\nif __name__ == '__main__':\n    raise SystemExit(5)\n"
        assert self.run(code)["outcome"] == "pass"

    def test_stdin_is_closed(self):
        code = "def add(a, b):\n    return a + b\ntry:\n    input()\nexcept EOFError:\n    pass\n"
        assert self.run(code)["outcome"] == "pass"

    def test_no_code_is_reported(self):
        assert self.run(None)["outcome"] == "no_code"
        assert self.run("   ")["outcome"] == "no_code"

    def test_class_based_tests_share_the_namespace(self):
        code = "class Acc:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n"
        tests = ["a = Acc(); a.inc(); assert a.n == 1", "assert Acc().n == 0"]
        assert self.run(code, tests)["outcome"] == "pass"

    def test_run_many_keeps_order_and_isolates_candidates(self):
        jobs = [(GOOD, TESTS), ("while True: pass", TESTS), ("def add(a, b):\n    return 0\n", TESTS), (None, TESTS)]
        out = E4.run_many(jobs, allow_exec=True, workers=3, timeout_s=1.0)
        assert [o["outcome"] for o in out] == ["pass", "timeout", "fail", "no_code"]


class TestAggregation:
    def test_pass_at_1_and_outcome_counts(self):
        def mk(o):
            return {"outcome": o, "tests_passed": 0, "tests_total": 1}

        s = E4.summarize_outcomes([mk("pass"), mk("pass"), mk("fail"), mk("timeout")])
        assert s["pass_at_1"] == 0.5 and s["passed"] == 2 and s["n"] == 4 and s["executed"] is True
        assert s["outcomes"] == {"fail": 1, "pass": 2, "timeout": 1}

    def test_empty(self):
        assert E4.summarize_outcomes([])["pass_at_1"] is None


# --------------------------------------------------------------- statistics
def items(correct, loss_per_token=1.0, tokens=10):
    return [{"qid": i, "correct": bool(c), "answer_loss_sum": loss_per_token * tokens, "answer_tokens": tokens}
            for i, c in enumerate(correct)]


class TestStatistics:
    def test_wilson_matches_the_textbook_value(self):
        assert E4.wilson(50, 100) == [0.4038, 0.5962]
        assert E4.wilson(0, 0) == [0.0, 1.0]

    def test_mcnemar_exact_known_value(self):
        a = [True] * 9 + [False] * 1 + [True] * 5
        b = [False] * 9 + [True] * 1 + [True] * 5
        r = E4.mcnemar_exact(a, b)
        assert (r["a_only"], r["b_only"]) == (9, 1)
        assert r["p"] == pytest.approx(2 * (1 + 10) / 1024, abs=1e-5)

    def test_mcnemar_without_discordant_pairs(self):
        assert E4.mcnemar_exact([True, False], [True, False]) == {"a_only": 0, "b_only": 0, "p": 1.0}

    def test_mcnemar_is_symmetric_in_p(self):
        a, b = [True] * 8 + [False] * 2, [False] * 8 + [True] * 2
        assert E4.mcnemar_exact(a, b)["p"] == E4.mcnemar_exact(b, a)["p"]

    def test_paired_bootstrap_on_a_clean_difference(self):
        n = 40
        r = E4.paired_bootstrap([1] * n, [0] * n, [8.0] * n, [10.0] * n, [10] * n, [10] * n, n_boot=200)
        assert r["acc_diff"] == 1.0 and r["acc_diff_ci95"] == [1.0, 1.0]
        assert r["loss_diff"] == pytest.approx(-0.2) and r["loss_diff_ci95"] == [-0.2, -0.2]

    def test_bootstrap_is_reproducible(self):
        rng = np.random.default_rng(3)
        a, b = rng.integers(0, 2, 60), rng.integers(0, 2, 60)
        loss = rng.uniform(5, 15, 60)
        x = E4.paired_bootstrap(a, b, loss, loss + 1, [10] * 60, [10] * 60, n_boot=500)
        y = E4.paired_bootstrap(a, b, loss, loss + 1, [10] * 60, [10] * 60, n_boot=500)
        assert x == y

    def test_majority_correct_needs_strictly_more_than_half(self):
        v = E4.majority_correct([[1, 1, 0, 0], [1, 0, 1, 0]])  # two seeds
        assert list(v) == [True, False, False, False]
        v3 = E4.majority_correct([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
        assert list(v3) == [True, True, False]
        assert list(E4.majority_correct([[1, 0]])) == [True, False]

    def test_pair_stats_pools_and_votes(self):
        n = 100
        rng = np.random.default_rng(0)
        base = rng.random(n) < 0.5
        a_seeds = {s: items(np.where(rng.random(n) < 0.9, base, ~base), loss_per_token=0.7) for s in (0, 1, 2)}
        b_seeds = {s: items(np.where(rng.random(n) < 0.9, base, ~base), loss_per_token=0.9) for s in (0, 1, 2)}
        ps = E4.pair_stats(a_seeds, b_seeds, n_boot=300)
        assert ps["seeds_a"] == [0, 1, 2] and ps["n_items"] == n
        assert ps["pooled_over_seeds"]["loss_diff"] == pytest.approx(-0.2, abs=1e-3)
        assert ps["pooled_over_seeds"]["loss_diff_ci95"][1] < 0
        assert len(ps["per_seed"]) == 3
        assert {"acc_a", "acc_b", "mcnemar"} <= set(ps["majority_vote"])

    def test_pair_stats_rejects_mismatched_items(self):
        a = {0: items([1, 0, 1])}
        b = {0: [dict(x, qid=x["qid"] + 100) for x in items([1, 0, 1])]}
        with pytest.raises(ValueError, match="same qids"):
            E4.pair_stats(a, b, n_boot=10)

    def test_pair_stats_pairs_seeds_and_falls_back_to_the_first(self):
        a = {0: items([1, 1, 0, 0]), 5: items([1, 0, 0, 0])}
        b = {0: items([0, 0, 0, 0])}
        ps = E4.pair_stats(a, b, n_boot=10)
        assert [(p["seed_a"], p["seed_b"]) for p in ps["per_seed"]] == [(0, 0), (5, 0)]


# ---------------------------------------------------------- decision rules
def pair(loss_diff=0.0, a_only=0, b_only=0, p=1.0, ci=(-0.1, 0.1)):
    return {"pooled_over_seeds": {"loss_diff": loss_diff, "loss_diff_ci95": list(ci), "acc_diff": 0.0,
                                  "acc_diff_ci95": [0.0, 0.0]},
            "majority_vote": {"acc_a": 0.5, "acc_b": 0.5,
                              "mcnemar": {"a_only": a_only, "b_only": b_only, "p": p}}}


class TestPrecheck:
    @pytest.mark.parametrize("p,ok", [
        (0.10, False), (0.1001, True), (0.35, True), (0.7999, True), (0.80, False), (0.0, False),
        (1.0, False), (0.05, False), (0.95, False)])
    def test_pass_at_1_must_be_strictly_inside(self, p, ok):
        v = E4.precheck_verdict(p, 0.8)
        assert v["ok"] is ok and v["verdict"] == ("PROCEED" if ok else "ABORT")

    def test_unmeasured_pass_rate_aborts(self):
        assert E4.precheck_verdict(None, 0.8)["verdict"] == "ABORT"
        assert E4.precheck_verdict(float("nan"), 0.8)["verdict"] == "ABORT"

    @pytest.mark.parametrize("loss", [None, float("nan"), float("inf")])
    def test_loss_must_be_recorded(self, loss):
        v = E4.precheck_verdict(0.4, loss)
        assert v["ok"] is False and any("loss" in r for r in v["reasons"])


class TestBeatsRule:
    @pytest.mark.parametrize("loss_diff,expect", [
        (-0.06, True), (-0.05, True), (-0.0501, True), (-0.0499, False), (0.0, False), (0.10, False)])
    def test_loss_criterion_threshold_is_0_05_nats(self, loss_diff, expect):
        r = E4.beats(pair(loss_diff=loss_diff))
        assert r["beats"] is expect and r["loss_criterion"] is expect

    @pytest.mark.parametrize("a_only,b_only,p,expect", [
        (20, 5, 0.004, True), (20, 5, 0.0499, True), (20, 5, 0.05, False), (12, 6, 0.2, False),
        (5, 20, 0.004, False),  # significant, but the wrong way round
        (0, 0, 1.0, False)])
    def test_accuracy_criterion_needs_direction_and_p_below_0_05(self, a_only, b_only, p, expect):
        r = E4.beats(pair(loss_diff=-0.01, a_only=a_only, b_only=b_only, p=p))
        assert r["accuracy_criterion"] is expect and r["beats"] is expect

    def test_either_criterion_is_enough(self):
        both = E4.beats(pair(loss_diff=-0.2, a_only=30, b_only=2, p=1e-6))
        assert both["loss_criterion"] and both["accuracy_criterion"] and both["beats"]

    def test_reports_whether_the_loss_interval_excludes_zero(self):
        assert E4.beats(pair(-0.06, ci=(-0.08, -0.04)))["loss_ci_excludes_zero"] is True
        assert E4.beats(pair(-0.06, ci=(-0.08, 0.01)))["loss_ci_excludes_zero"] is False


class TestVerdicts:
    def test_premise_passes_when_full_ft_clearly_better(self):
        v = E4.premise_verdict(pair(loss_diff=-0.07), 3, 3)
        assert v["verdict"] == "PASS" and v["proceed"] is True
        assert v["gate"] == "premise_3b_full_ft_beats_qlora"

    def test_premise_fails_when_qlora_is_as_good(self):
        v = E4.premise_verdict(pair(loss_diff=-0.02, a_only=10, b_only=9, p=1.0), 3, 3)
        assert v["verdict"] == "FAIL" and v["proceed"] is False

    def test_premise_fails_when_qlora_is_better(self):
        v = E4.premise_verdict(pair(loss_diff=+0.2, a_only=3, b_only=30, p=1e-6), 3, 3)
        assert v["verdict"] == "FAIL"

    @pytest.mark.parametrize("sa,sb", [(1, 3), (3, 1), (0, 0), (1, 1)])
    def test_too_few_seeds_is_inconclusive_and_does_not_proceed(self, sa, sb):
        v = E4.premise_verdict(pair(loss_diff=-0.5), sa, sb)
        assert v["verdict"] == "INCONCLUSIVE" and v["proceed"] is False

    def test_missing_pair_is_inconclusive(self):
        assert E4.premise_verdict(None, 3, 3)["verdict"] == "INCONCLUSIVE"

    def test_two_seeds_are_enough(self):
        assert E4.premise_verdict(pair(loss_diff=-0.06), 2, 2)["verdict"] == "PASS"

    def test_ship_rule_uses_the_same_rule_under_its_own_name(self):
        ok = E4.ship_verdict(pair(loss_diff=-0.06), 2, 2)
        no = E4.ship_verdict(pair(loss_diff=0.052), 2, 2)  # the stage-d GSM8K result: +0.052, a drop
        assert ok["verdict"] == "PASS" and ok["gate"] == "ship_7b_engine_b_beats_qlora"
        assert no["verdict"] == "FAIL"

    def test_verdict_is_computed_from_real_pair_stats(self):
        n = 120
        rng = np.random.default_rng(1)
        solved = rng.random(n) < 0.5
        full = {s: items(solved, loss_per_token=0.60) for s in (0, 1, 2)}
        qlora = {s: items(solved, loss_per_token=0.70) for s in (0, 1, 2)}
        better = E4.pair_stats(full, qlora, n_boot=200)
        assert E4.premise_verdict(better, 3, 3)["verdict"] == "PASS"
        same = E4.pair_stats(full, {s: items(solved, loss_per_token=0.61) for s in (0, 1, 2)}, n_boot=200)
        assert E4.premise_verdict(same, 3, 3)["verdict"] == "FAIL"


# ------------------------------------------------------------------- budget
class TestBudget:
    def spec(self, arm="default", seed=0, size="3b"):
        return E4.RunSpec(size, arm, seed)

    def test_plan_order_3b(self):
        plan = E4.plan_for("3b")
        assert [(s.arm, s.seed) for s in plan] == [
            ("default", 0), ("qlora", 0), ("default", 1), ("qlora", 1), ("default", 2), ("qlora", 2),
            ("block_k5", 0), ("block_k5", 1), ("block_k5", 2)]
        assert plan[0].tag == "e4_3b_default_s0"

    def test_plan_order_7b_puts_galore_last(self):
        plan = E4.plan_for("7b")
        assert [(s.arm, s.seed) for s in plan] == [
            ("block_k5", 0), ("qlora", 0), ("block_k5", 1), ("qlora", 1), ("galore", 0)]

    def test_unknown_size(self):
        with pytest.raises(ValueError):
            E4.plan_for("13b")

    def test_deadline_is_cap_over_rate_minus_reserve(self):
        g = E4.BudgetGuard(cap_usd=4.0, rate_usd_h=0.9, start_epoch=1000.0, reserve_s=600.0)
        assert g.deadline_epoch == pytest.approx(1000.0 + 4.0 / 0.9 * 3600 - 600.0)

    def test_admits_what_fits_and_drops_what_does_not(self):
        g = E4.BudgetGuard(cap_usd=1.0, rate_usd_h=1.0, start_epoch=0.0, reserve_s=0.0)  # 3600 s
        assert g.admit(self.spec(), 3000, now=0.0)[0] is True
        ok, why = g.admit(self.spec(seed=1), 700, now=3000.0)
        assert ok is False and "past the deadline" in why
        assert g.dropped[0]["tag"] == "e4_3b_default_s1"

    def test_a_drop_is_sticky_so_priority_is_never_inverted(self):
        g = E4.BudgetGuard(cap_usd=1.0, rate_usd_h=1.0, start_epoch=0.0, reserve_s=0.0)
        assert g.admit(self.spec(), 4000, now=0.0)[0] is False
        ok, why = g.admit(self.spec("block_k5"), 10, now=0.0)  # tiny, would fit on its own
        assert ok is False and "sticky" in why

    def test_walk_plan_drops_engine_b_first_in_a_tight_budget(self):
        plan = E4.plan_for("3b")

        def est(s):
            return 3000.0 if s.arm != "block_k5" else 900.0

        g = E4.BudgetGuard(cap_usd=4.0, rate_usd_h=0.9, start_epoch=0.0, reserve_s=600.0)
        admitted = E4.walk_plan(plan, g, est, now=0.0)
        kept = {(s.arm, s.seed) for s in admitted}
        assert {("default", 0), ("qlora", 0), ("default", 1), ("qlora", 1)} <= kept
        assert [d["tag"] for d in g.dropped][-3:] == [
            "e4_3b_block_k5_s0", "e4_3b_block_k5_s1", "e4_3b_block_k5_s2"]

    def test_walk_plan_with_a_generous_budget_runs_everything(self):
        plan = E4.plan_for("7b")
        g = E4.BudgetGuard(cap_usd=100.0, start_epoch=0.0)
        assert E4.walk_plan(plan, g, lambda s: 600.0, now=0.0) == plan and g.dropped == []

    def test_estimate_uses_stage_d_s_per_step(self):
        s = E4.RunSpec("3b", "qlora", 0)
        est = E4.estimate_run_s(s, 5000, eval_s=300, load_s=60, overhead_s=30, margin=1.0)
        assert est == pytest.approx(5000 * 0.434 + 390)
        assert E4.estimate_run_s(s, 5000, eval_s=0, measured_wall_s=1000.0, margin=1.5) == 1500.0

    def test_stage_d_numbers_are_the_documented_ones(self):
        assert E4.S_PER_STEP == {("3b", "default"): 0.317, ("3b", "qlora"): 0.434, ("3b", "block_k5"): 0.144,
                                 ("7b", "block_k5"): 0.220, ("7b", "qlora"): 0.534, ("7b", "galore"): 1.271}


# ------------------------------------------- stage summary and pod orchestrator
def write_run(out: Path, size: str, arm: str, seed: int, correct, loss_per_token, *, status="ok"):
    tag = f"e4_{size}_{arm}_s{seed}"
    (out / "runs" / "items").mkdir(parents=True, exist_ok=True)
    rel = f"runs/items/{tag}.jsonl.gz"
    with gzip.open(out / rel, "wt", encoding="utf-8") as fh:
        for it in items(correct, loss_per_token):
            fh.write(json.dumps(it) + "\n")
    n = len(correct)
    rec = {"tag": tag, "arm": arm, "model": "Qwen/Qwen2.5-3B", "seed": seed, "status": status, "n": n,
           "pass_at_1": float(np.mean(correct)), "acc_wilson95": [0.0, 1.0],
           "heldout_after": loss_per_token, "heldout_before": 1.5, "pass_at_1_before": 0.4,
           "s_per_step": 0.3, "wall_s": 100.0, "eval_s": 20.0, "items_file": rel, "steps": 5000,
           "outcomes": {"pass": int(np.sum(correct)), "fail": int(n - np.sum(correct))},
           "mask_check": {"input_sha256": "aa", "mask_sha256": "bb", "loss_tokens": 99}}
    (out / "runs" / f"{tag}.json").write_text(json.dumps(rec), encoding="utf-8")


def populate(out: Path, *, full_loss, qlora_loss, seeds=(0, 1, 2), n=80, size="3b",
             a_arm="default", b_arm="qlora"):
    rng = np.random.default_rng(7)
    solved = rng.random(n) < 0.5
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for s in seeds:
        write_run(out, size, a_arm, s, solved, full_loss)
        write_run(out, size, b_arm, s, solved, qlora_loss)


class TestStageSummary:
    def test_premise_pass_written_to_the_stage_file(self, tmp_path):
        populate(tmp_path, full_loss=0.55, qlora_loss=0.65)
        rec = pod_e4_summary.summarize_stage(str(tmp_path), "3b", n_boot=200)
        assert rec["verdict"]["verdict"] == "PASS"
        assert rec["pairs"]["default|qlora"]["pooled_over_seeds"]["loss_diff"] == pytest.approx(-0.1)
        assert rec["arms"]["default"]["seeds"] == [0, 1, 2]
        assert all(rec["checks"].values())

    def test_premise_fail(self, tmp_path):
        populate(tmp_path, full_loss=0.62, qlora_loss=0.65)
        assert pod_e4_summary.summarize_stage(str(tmp_path), "3b", n_boot=200)["verdict"]["verdict"] == "FAIL"

    def test_inconclusive_with_one_seed_per_arm(self, tmp_path):
        populate(tmp_path, full_loss=0.5, qlora_loss=0.9, seeds=(0,))
        assert pod_e4_summary.summarize_stage(str(tmp_path), "3b", n_boot=200)["verdict"]["verdict"] \
            == "INCONCLUSIVE"

    def test_failed_runs_do_not_count_as_seeds(self, tmp_path):
        populate(tmp_path, full_loss=0.5, qlora_loss=0.9, seeds=(0, 1))
        write_run(tmp_path, "3b", "default", 2, [1] * 80, 0.1, status="error")
        rec = pod_e4_summary.summarize_stage(str(tmp_path), "3b", n_boot=200)
        assert rec["arms"]["default"]["seeds"] == [0, 1]
        assert rec["checks"]["completed:e4_3b_default_s2"] is False

    def test_seven_b_stage_uses_the_ship_rule(self, tmp_path):
        populate(tmp_path, size="7b", full_loss=0.6, qlora_loss=0.55, seeds=(0, 1),
                 a_arm="block_k5", b_arm="qlora")
        rec = pod_e4_summary.summarize_stage(str(tmp_path), "7b", n_boot=200)
        assert rec["verdict"]["gate"] == "ship_7b_engine_b_beats_qlora"
        assert rec["verdict"]["verdict"] == "FAIL"

    def test_mask_mismatch_is_flagged(self, tmp_path):
        populate(tmp_path, full_loss=0.5, qlora_loss=0.6, seeds=(0,))
        p = tmp_path / "runs" / "e4_3b_qlora_s0.json"
        rec = json.loads(p.read_text())
        rec["mask_check"]["mask_sha256"] = "zz"
        p.write_text(json.dumps(rec))
        out = pod_e4_summary.summarize_stage(str(tmp_path), "3b", n_boot=50)
        assert out["checks"]["same_first_batch_and_mask:s0"] is False

    def test_cli_exit_codes(self, tmp_path):
        populate(tmp_path, full_loss=0.55, qlora_loss=0.65)
        assert pod_e4_summary.main(["--out", str(tmp_path), "--size", "3b", "--boot", "100"]) == 0
        assert (tmp_path / "stage_e4_3b.json").exists()
        populate(tmp_path / "b", full_loss=0.65, qlora_loss=0.65)
        assert pod_e4_summary.main(["--out", str(tmp_path / "b"), "--size", "3b", "--boot", "100"]) == 3
        assert pod_e4_summary.main(["--out", str(tmp_path / "missing"), "--size", "3b"]) == 2


class TestOrchestrator:
    def cfg(self, tmp_path, size="3b", **kw):
        base = {"out": str(tmp_path), "size": size, "model": "m", "steps": 5000, "batch": 4, "seq": 512,
                "seeds": (0, 1, 2), "cap_usd": 4.0, "rate_usd_h": 0.9, "reserve_s": 600.0, "n_eval": 500,
                "max_new_tokens": 768}
        base.update(kw)
        (tmp_path / "runs").mkdir(exist_ok=True)
        return pod_e4.Cfg(**base)

    def test_arm_arguments_match_stage_d(self):
        sh = (SCRIPTS / "pod_block_engine.sh").read_text(encoding="utf-8")
        for arm in ("default", "block_k5", "qlora", "galore"):
            line = next(ln for ln in sh.splitlines() if ln.strip().startswith(f"{arm})") and "echo" in ln)
            tail = line.split("--model $M ", 1)[1].rstrip('" ;')
            assert " ".join(pod_e4.ARM_ARGS[arm]) in tail, (arm, tail)

    def test_train_args_pin_seed_steps_and_shape(self, tmp_path):
        args = pod_e4.train_args(self.cfg(tmp_path), E4.RunSpec("3b", "qlora", 2))
        assert args[:2] == ["--tag", "e4_3b_qlora_s2"]
        i = args.index("--steps")
        assert args[i:i + 2] == ["--steps", "5000"]
        assert {"--qlora", "--lr-default", "--seed"} <= set(args)

    def test_seven_b_refuses_without_a_passing_premise_verdict(self, tmp_path):
        cfg = self.cfg(tmp_path, size="7b", seeds=(0, 1), cap_usd=4.5)
        (tmp_path / "precheck_7b.json").write_text(json.dumps({"ok": True}))
        assert pod_e4._require_gates(cfg) == pod_e4.EXIT_GATE_STOP
        for verdict in ("FAIL", "INCONCLUSIVE"):
            (tmp_path / "stage_e4_3b.json").write_text(json.dumps({"verdict": {"verdict": verdict}}))
            assert pod_e4._require_gates(cfg) == pod_e4.EXIT_GATE_STOP
        (tmp_path / "stage_e4_3b.json").write_text(json.dumps({"verdict": {"verdict": "PASS"}}))
        assert pod_e4._require_gates(cfg) is None

    def test_premise_override_is_explicit(self, tmp_path, monkeypatch):
        cfg = self.cfg(tmp_path, size="7b", seeds=(0, 1), cap_usd=4.5)
        (tmp_path / "precheck_7b.json").write_text(json.dumps({"ok": True}))
        monkeypatch.setenv("E4_PREMISE_OVERRIDE", "decided in review")
        assert pod_e4._require_gates(cfg) is None

    def test_training_refuses_without_a_passing_precheck(self, tmp_path):
        cfg = self.cfg(tmp_path)
        assert pod_e4._require_gates(cfg) == pod_e4.EXIT_PRECHECK_ABORT
        (tmp_path / "precheck_3b.json").write_text(json.dumps({"ok": False, "verdict": "ABORT"}))
        assert pod_e4._require_gates(cfg) == pod_e4.EXIT_PRECHECK_ABORT
        (tmp_path / "precheck_3b.json").write_text(json.dumps({"ok": True}))
        assert pod_e4._require_gates(cfg) is None

    def test_dry_run_skips_the_gates(self, tmp_path):
        assert pod_e4._require_gates(self.cfg(tmp_path, dry_run=True)) is None

    def test_defaults_are_the_pre_registered_ones(self):
        assert pod_e4.CAPS_USD == {"3b": 4.80, "7b": 4.50}  # 3B amended by the Director, 2026-10-01
        assert pod_e4.MODELS == {"3b": "Qwen/Qwen2.5-3B", "7b": "Qwen/Qwen2.5-7B"}

    def test_stage_skips_existing_receipts_and_drops_the_tail_when_the_budget_runs_out(
            self, tmp_path, monkeypatch):
        cfg = self.cfg(tmp_path, seeds=(0, 1), steps=100, cap_usd=0.5, start_epoch=1000.0)
        # deadline = 1000 + 0.5 / 0.9 h - 600 s = 2400
        (tmp_path / "precheck_3b.json").write_text(json.dumps({"ok": True}))
        (tmp_path / "runs" / "e4_3b_default_s0.json").write_text(json.dumps({"status": "ok", "wall_s": 10}))
        clock = {"t": 1000.0}
        launched: list[str] = []

        def fake_driver(c, mode, extra, env_extra=None):
            tag = extra[1]
            launched.append(tag)
            clock["t"] += 800.0  # every run "takes" 800 s
            (tmp_path / "runs" / f"{tag}.json").write_text(json.dumps(
                {"status": "ok", "wall_s": 800.0, "pass_at_1": 0.5, "heldout_after": 0.5}))
            return 0

        monkeypatch.setattr(pod_e4, "driver", fake_driver)
        monkeypatch.setattr(pod_e4, "run_py", lambda *a, **k: 0)
        monkeypatch.setattr(pod_e4.time, "time", lambda: clock["t"])
        assert pod_e4.cmd_stage(cfg) == 0
        assert "e4_3b_default_s0" not in launched  # receipt exists: resumed, not re-run
        assert launched == ["e4_3b_qlora_s0", "e4_3b_default_s1"]
        budget = json.loads((tmp_path / "budget_3b.json").read_text())
        assert [d["tag"] for d in budget["dropped"]] == [
            "e4_3b_qlora_s1", "e4_3b_block_k5_s0", "e4_3b_block_k5_s1"]  # sticky: the tail goes together
        assert budget["drop_order"][0] == "e4_3b_block_k5_s1"


def test_driver_exposes_the_code_dataset_and_the_exec_flag():
    """The driver is a script (argparse at import), so only its source is checked here."""
    src = (SCRIPTS / "pod_block_engine.py").read_text(encoding="utf-8")
    assert 'choices=["dolly", "gsm8k", "code"]' in src
    assert "--allow-exec" in src and "E4.run_many(jobs, allow_exec=True" in src
