# Calibration safety (external review 2026-10-02, calibration lane).
"""One test per finding: the probe cap, the orphaned child, the store, the
range a measurement covers, and full fine-tuning's fixed cost.

The probe's optimizer is pinned in the GPU-only child; here it is pinned as a
constant, and ``tests/test_vram_calibration_gpu.py`` checks the stored value
on real hardware.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from backpropagate import vram_calibration as vc
from backpropagate.trainer import estimate_vram

GIB = vc.GIB
MACHINE = {"gpu": "Test GPU", "vram_gib": 24.0, "torch": "2.10", "transformers": "5.5",
           "unsloth": "2026.5", "attention": "sdpa-efficient"}
_SHAPE = {"param_count_billions": 1.236, "hidden_dim": 2048, "num_layers": 16,
          "num_heads": 32, "vocab_size": 128256}
_QV = {"target_modules": "q_proj,v_proj", "lora_r": 16}


def _cal(**kw) -> vc.Calibration:
    base = {
        "model": "org/m-1B", "mode": "lora", "base_4bit": True, "machine": MACHINE,
        "load_gib": 1.0, "floor_gib": 1.0, "quad_bytes": 0.0, "lin_bytes": 550_000.0,
        "probe_trainable_params": 850_000, "max_residual_pct": 1.0,
        "seq_min": 1024, "seq_max": 2048, "optim": "adamw_8bit",
    }
    base.update(kw)
    return vc.Calibration(**base)


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    monkeypatch.setenv("BACKPROPAGATE_VRAM_CALIBRATION", str(path))
    return path


# ---- the cap never exceeds what is free -----------------------------------------------


@pytest.mark.parametrize(("free", "total"), [(2.1, 32.0), (2.1, 80.0), (3.9, 32.0), (0.5, 8.0)])
def test_too_little_free_memory_runs_nothing(free, total):
    assert vc.probe_budget(free, total) is None


@pytest.mark.parametrize(
    ("free", "total"),
    [(30.5, 32.0), (8.0, 8.0), (3.1, 8.0), (12.0, 16.0), (79.0, 80.0), (20.0, 80.0), (4.2, 32.0)],
)
def test_the_cap_stays_below_free_minus_headroom(free, total):
    budget, fraction = vc.probe_budget(free, total)
    headroom = max(vc.HEADROOM_MIN_GIB, vc.HEADROOM_SHARE * total)
    assert budget == pytest.approx(free - headroom)
    assert budget >= vc.MIN_BUDGET_GIB
    assert fraction * total <= free - headroom + 1e-9
    assert 0 < fraction <= vc.MAX_FRACTION


def test_the_cap_is_never_raised_to_a_floor():
    # The old rule was max(0.05, budget / total): 2.1 GiB free on a 32 GB card
    # gave a 1.6 GiB cap with 1.1 GiB of budget.
    assert vc.probe_budget(2.1, 32.0) is None
    budget, fraction = vc.probe_budget(4.2, 32.0)
    assert fraction == pytest.approx(budget / 32.0) and fraction < 0.06


def test_the_probe_optimizer_is_not_a_paged_one():
    assert vc.PROBE_OPTIM == "adamw_8bit"
    assert not vc.PROBE_OPTIM.startswith("paged")


# ---- the child does not outlive the parent ---------------------------------------------


class _Proc:
    """A stand-in for Popen: ``lines`` then EOF; alive until killed."""

    def __init__(self, lines, hold: float = 0.0):
        self._lines = lines
        self._hold = hold
        self.killed = threading.Event()
        self.stdout = self

    def __iter__(self):
        yield from (line.encode("utf-8") + b"\n" for line in self._lines)
        if self._hold:
            self.killed.wait(self._hold)

    def wait(self):
        return 0

    def poll(self):
        return -9 if self.killed.is_set() else None

    def kill(self):
        self.killed.set()


def _result_line(probes=(), **extra):
    row = {"event": "result", "load_gib": 1.06, "floor_gib": 0.98, "trainable_params": 851_968,
           "formula_quad": 0.0, "formula_lin": 574_464.0, "probes": list(probes),
           "optim": "adamw_8bit"}
    row.update(extra)
    return "CALIBRATION " + json.dumps(row)


def _patch_popen(monkeypatch, proc):
    seen: dict = {}

    def popen(argv, **kwargs):
        seen.update(kwargs)
        return proc

    monkeypatch.setattr(vc, "machine_fingerprint", lambda: MACHINE)
    monkeypatch.setattr(vc.subprocess, "Popen", popen)
    return seen


def test_the_child_is_killed_when_the_parent_stops_listening(monkeypatch, store):
    proc = _Proc(['CALIBRATION {"event": "loading", "budget_gib": 20.0}'], hold=5.0)
    _patch_popen(monkeypatch, proc)

    def on_event(row):
        raise KeyboardInterrupt  # Ctrl+C while the probes run

    with pytest.raises(KeyboardInterrupt):
        vc.calibrate("org/m-1B", on_event=on_event)
    assert proc.killed.is_set()
    assert not store.exists()


def test_the_child_gets_its_own_session_on_posix(monkeypatch, store):
    seen = _patch_popen(monkeypatch, _Proc([_result_line()]))
    vc.calibrate("org/m-1B")
    assert seen.get("start_new_session", False) is (os.name != "nt")


def test_a_result_that_arrives_as_the_timer_fires_is_kept(monkeypatch, store):
    probes = [{"batch": 2, "seq": 2048, "overhead_gib": 2.12}]
    proc = _Proc([_result_line(probes)], hold=5.0)  # result printed, then the child lingers
    _patch_popen(monkeypatch, proc)
    cal = vc.calibrate("org/m-1B", timeout_s=0.2)
    assert proc.killed.is_set()  # the timer stopped it
    assert cal.rows_measured and cal.seq_max == 2048 and cal.optim == "adamw_8bit"
    assert store.exists()


def test_no_lease_without_a_process_handle():
    assert vc._kill_on_close(_Proc([])) is None
    vc._release(None)  # nothing to close


def test_kill_tree_never_signals_a_made_up_pid(monkeypatch):
    called = []
    monkeypatch.setattr(vc.subprocess, "run", lambda *a, **k: called.append(a))
    proc = _Proc([])
    vc._kill_tree(proc)
    assert called == [] and proc.killed.is_set()


# ---- the store -------------------------------------------------------------------------


def test_an_unreadable_store_is_moved_aside_not_overwritten(store, caplog):
    store.write_text('{"half a row": ', encoding="utf-8")
    assert vc.lookup("org/m-1B", machine=MACHINE) is None  # readers see an empty store
    vc.save(_cal())
    aside = store.with_name(store.name + ".unreadable")
    assert aside.read_text(encoding="utf-8") == '{"half a row": '
    assert vc.lookup("org/m-1B", machine=MACHINE) is not None
    assert any("moved to" in r.getMessage() for r in caplog.records)


def test_a_store_that_is_not_an_object_is_moved_aside(store):
    store.write_text("[1, 2, 3]", encoding="utf-8")
    vc.save(_cal())
    assert store.with_name(store.name + ".unreadable").exists()
    assert len(json.loads(store.read_text(encoding="utf-8"))) == 1


def test_save_keeps_other_rows_and_leaves_no_temp_or_lock_files(store):
    vc.save(_cal(model="org/a-1B"))
    vc.save(_cal(model="org/b-1B"))
    assert len(json.loads(store.read_text(encoding="utf-8"))) == 2
    assert [p.name for p in store.parent.iterdir()] == [store.name]


def test_a_lock_left_by_a_dead_writer_is_taken_over(store, monkeypatch):
    lock = store.with_name(store.name + ".lock")
    lock.write_text("", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    vc.save(_cal())
    assert store.exists() and not lock.exists()


def test_a_live_lock_is_waited_for_then_the_write_goes_ahead(store, monkeypatch):
    monkeypatch.setattr(vc, "_LOCK_WAIT_S", 0.2)
    lock = store.with_name(store.name + ".lock")
    lock.write_text("", encoding="utf-8")  # another writer, still working
    started = time.monotonic()
    vc.save(_cal())
    assert time.monotonic() - started >= 0.2
    assert store.exists()
    assert lock.exists()  # not ours to remove


@pytest.mark.parametrize(
    "bad",
    [
        {"load_gib": float("nan")},
        {"load_gib": -1.0},
        {"load_gib": 0.0},
        {"floor_gib": -0.5},
        {"lin_bytes": -5.0},
        {"lin_bytes": float("inf")},
        {"fixed_gib": -1.0},
        {"load_gib": True},
        {"seq_max": -2048},
        {"probe_trainable_params": "many"},
    ],
)
def test_a_stored_row_with_impossible_numbers_is_ignored(store, bad):
    vc.save(_cal())
    data = json.loads(store.read_text(encoding="utf-8"))
    for entry in data.values():
        entry.update(bad)
    store.write_text(json.dumps(data), encoding="utf-8")
    assert vc.lookup("org/m-1B", machine=MACHINE) is None


def test_a_row_from_the_previous_format_is_ignored(store):
    vc.save(_cal())
    data = json.loads(store.read_text(encoding="utf-8"))
    for entry in data.values():
        entry["version"] = 2
    store.write_text(json.dumps(data), encoding="utf-8")
    assert vc.lookup("org/m-1B", machine=MACHINE) is None


# ---- what a measurement covers ---------------------------------------------------------


def _use(monkeypatch, cal):
    monkeypatch.setattr(
        vc, "lookup", lambda model, mode="lora", base_4bit=True, machine=None: cal
    )


def test_a_measurement_is_used_within_twice_its_longest_probe(monkeypatch):
    _use(monkeypatch, _cal())
    for seq in (512, 2048, 4096):
        est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=2, max_seq_length=seq)
        assert est.source == "measured", seq


def test_rows_far_longer_than_the_probes_fall_back_to_the_formula(monkeypatch):
    _use(monkeypatch, _cal())
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=1, max_seq_length=8192)
    assert est.source == "estimate"
    assert any("probed rows up to 2048 tokens" in n for n in est.notes)


def test_a_row_without_a_recorded_range_is_still_used(monkeypatch):
    # No informative probe ran (a big model on a small card): load size only.
    _use(monkeypatch, _cal(quad_bytes=None, lin_bytes=None, seq_min=0, seq_max=0))
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=1, max_seq_length=8192)
    assert est.source == "measured"


@pytest.mark.parametrize("method", ["orpo", "simpo", "kto", "ORPO"])
def test_preference_methods_never_read_measured(monkeypatch, method):
    _use(monkeypatch, _cal())
    est = estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=2, method=method)
    assert est.source == "estimate"
    assert any("two sequences per example" in n for n in est.notes)


def test_sft_is_the_default_method(monkeypatch):
    _use(monkeypatch, _cal())
    assert estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=2, method="sft").source == "measured"
    assert estimate_vram("org/m-1B", **_SHAPE, **_QV, batch_size=2).source == "measured"


# ---- full fine-tuning: the fixed cost is measured once, not added twice -----------------


def test_fit_fixed_is_what_is_left_after_the_rows():
    per_token = 574_464.0
    fixed = 6.2
    probes = [
        {"batch": b, "seq": s, "overhead_gib": fixed + b * s * per_token / GIB}
        for b, s in ((2, 2048), (4, 2048), (4, 1024))
    ]
    assert vc.fit_fixed(probes, per_token) == pytest.approx(fixed)
    # A lighter probe (the compiled loss) never lowers it.
    probes.append({"batch": 4, "seq": 2048, "overhead_gib": fixed + 1.0})
    assert vc.fit_fixed(probes, per_token) == pytest.approx(fixed)


def test_fit_fixed_without_usable_probes_is_zero():
    assert vc.fit_fixed([], 500_000.0) == 0.0
    assert vc.fit_fixed([{"batch": 2, "seq": 2048, "oom": True}], 500_000.0) == 0.0
    # Rows priced above what was measured: never negative.
    assert vc.fit_fixed([{"batch": 2, "seq": 2048, "overhead_gib": 0.5}], 574_464.0) == 0.0


def test_a_full_fine_tune_calibration_stores_the_fixed_cost(monkeypatch, store):
    per_token = 574_464.0
    probes = [{"batch": 2, "seq": 2048, "overhead_gib": 6.2 + 2 * 2048 * per_token / GIB},
              {"batch": 4, "seq": 1024, "overhead_gib": 6.2 + 4 * 1024 * per_token / GIB}]
    proc = _Proc([_result_line(probes, load_gib=2.31, trainable_params=1_235_814_400)])
    _patch_popen(monkeypatch, proc)
    cal = vc.calibrate("org/m-1B", mode="full")
    assert cal.fixed_gib == pytest.approx(6.2, abs=1e-3)
    assert (cal.quad_bytes, cal.lin_bytes) == (0.0, per_token)
    assert cal.base_4bit is False
    # The estimate is load + fixed + rows: the formula's 5.3 bytes per
    # parameter is not added on top.
    monkeypatch.setattr(vc, "machine_fingerprint", lambda: MACHINE)
    est = estimate_vram("org/m-1B", mode="full", batch_size=2, overhead_fraction=0.0, **_SHAPE)
    assert est.source == "measured"
    assert est.total_gb == pytest.approx(2.31 + 6.2 + 2 * 2048 * per_token / GIB, abs=0.01)


def test_a_lora_calibration_has_no_fixed_cost(monkeypatch, store):
    probes = [{"batch": 2, "seq": 2048, "overhead_gib": 2.12}]
    _patch_popen(monkeypatch, _Proc([_result_line(probes)]))
    cal = vc.calibrate("org/m-1B")
    assert cal.fixed_gib == 0.0 and cal.lin_bytes == pytest.approx(2.12 * GIB / 4096)
