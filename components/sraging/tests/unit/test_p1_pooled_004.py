"""p1-pooled (tasks T111b; spec.md Clarifications 2026-11-01, 8f0f269): the start step extrapolated from P1's ladder,
K keys of a coupled pair pooled and judged against the n = 25,000 bar, at most two doublings then "not met"; nothing
depends on the worker count; one accessor for the frozen steps, p1-pooled's preferred; P3 and P2 refuse by name a
step that is not met. Tiny n here; the phase is never run in the unit suite. W <= 2."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

import acceptance_004 as ACC  # noqa: E402
import p1_pooled_004 as PP  # noqa: E402
from sraging import SRParams  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
BAR_CLAMP = 1.973 / math.sqrt(25_000) / 10


def test_the_start_step_is_extrapolated_from_the_finest_rung():
    rungs = [{"m": 16, "D": 0.004}, {"m": 64, "D": 0.0018}]
    ex = PP.extrapolate(rungs, 0.932, "clamp")
    b = 0.0018 / (2 ** 0.932 - 1)
    assert ex["m_f"] == 64 and ex["b_f"] == pytest.approx(b) and ex["bar"] == pytest.approx(BAR_CLAMP)
    assert ex["m_start"] == math.ceil(64 * (b / BAR_CLAMP) ** (1 / 0.932))
    assert PP.extrapolate([{"m": 16, "D": 0.0001}], 1.0, "clamp")["m_start"] == 16      # under the bar: m_f itself


def _cell(m_start=4):
    return {"id": "t|hom|clamp", "preset": "t", "het": "hom", "wall": "clamp", "base_dt": 0.5, "p_hat": 1.0,
            "m_start": m_start, "t_max": 300.0, "key": 7, "params": {k: v for k, v in P.as_dict().items()
                                                                       if k != "time_unit"},
            "het_spec": None, "n": 300}


def _fake_pooled(met_from):
    def f(cell, m, pool, k=PP.K):
        ok = m >= met_from
        return {"m": m, "dt": cell["base_dt"] / m, "K": k, "n_pooled": 0, "D_pooled": 0.0, "b": 0.0 if ok else 1.0,
                "bar": 0.5, "met": ok, "D_per_key": [], "noise_floor": 0.0, "noise_spread": None, "draw_cpu_s": 1.0,
                "steps_per_individual": 600.0 * m, "worker_cpu_s": 2.0}
    return f


def test_a_cell_over_the_bar_doubles_m_and_then_meets_it(monkeypatch):
    monkeypatch.setattr(PP, "pooled_at", _fake_pooled(met_from=8))
    rec, cpu = PP.run_cell(_cell(4), pool=None)
    assert [t["m"] for t in rec["tried"]] == [4, 8] and rec["met"] and rec["frozen"]["m"] == 8
    assert rec["frozen"]["dt"] == 0.5 / 8 and rec["status"] == "met" and cpu == 4.0
    twin, _ = PP.run_cell(_cell(8), pool=None)                                          # the twin: met at the start
    assert [t["m"] for t in twin["tried"]] == [8] and twin["frozen"]["m"] == 8


def test_a_cell_that_never_meets_the_bar_reads_not_met_after_two_doublings(monkeypatch):
    monkeypatch.setattr(PP, "pooled_at", _fake_pooled(met_from=10 ** 9))
    rec, _ = PP.run_cell(_cell(4), pool=None)
    assert [t["m"] for t in rec["tried"]] == [4, 8, 16] and not rec["met"] and rec["frozen"] is None
    assert rec["status"] == "not met"
    out = PP.result([rec])
    assert out["not_met"] == ["t|hom|clamp"] and out["frozen"]["t|hom|clamp"]["met"] is False


def _strip(r):
    return {k: v for k, v in r.items() if k not in ("draw_cpu_s",)}


def test_the_pooled_measurement_does_not_depend_on_the_worker_count():
    cell = _cell(1)
    with PP.Pool(1) as p1:
        a = PP.pooled_at(cell, 1, p1, k=3)
    with PP.Pool(2) as p2:
        b = PP.pooled_at(cell, 1, p2, k=3)
    a.pop("worker_cpu_s"), b.pop("worker_cpu_s")
    assert _strip(a) == _strip(b) and a["K"] == 3 and a["n_pooled"] == 900
    assert a["D_pooled"] > 0 and a["noise_floor"] == pytest.approx(np.mean(a["D_per_key"]) / math.sqrt(3))
    assert a["b"] == pytest.approx(a["D_pooled"] / (2 ** 1.0 - 1)) and a["bar"] == pytest.approx(
        1.973 / math.sqrt(300) / 10)


def _results(monkeypatch, pooled=True, met=True):
    p1 = {"status_for_acceptance": "acceptance", "steps": {"human|het|clamp": {"m": None, "dt": None, "met": False}},
          "steps_per_individual": {}, "draw_cpu_s": {}, "p_hat": {"human|het|clamp": 0.7}, "cells": [],
          "ns_per_individual_step": 12.0}
    pp = {"status_for_acceptance": "acceptance",
          "frozen": {"human|het|clamp": {"m": 97, "dt": 3.2e-5, "steps_per_individual": 2.3e6, "draw_cpu_s": 900.0,
                                         "p_hat": 0.7, "met": met},
                     "s5|het|clamp": {"m": 40, "dt": 3.1e-4, "steps_per_individual": 3.4e5, "draw_cpu_s": 60.0,
                                      "p_hat": 0.9, "met": met}}}
    files = {"p1-langevin": p1, **({"p1-pooled": pp} if pooled else {})}
    monkeypatch.setattr(ACC, "committed", lambda rel: any(rel == ACC.result_rel(k) for k in files))
    monkeypatch.setattr(ACC, "read_result", lambda phase: files[phase])
    return p1, pp


def test_the_accessor_prefers_p1_pooled_and_never_reads_p1s_steps_then(monkeypatch):
    _results(monkeypatch)
    fs = ACC.frozen_steps()
    assert fs.source == "p1-pooled" and fs["human|het|clamp"]["m"] == 97
    v = ACC.p1_steps()
    assert v["steps_source"] == "p1-pooled" and v["steps_per_individual"]["human|het|clamp"] == 2.3e6
    _results(monkeypatch, pooled=False)                                                 # the twin: P1's alone
    fs = ACC.frozen_steps()
    assert fs.source == "p1-langevin"
    with pytest.raises(ACC.Refused, match=r"human\|het\|clamp.*not met in p1-langevin"):
        fs["human|het|clamp"]


def test_the_accessor_refuses_a_cell_that_is_not_met_by_name(monkeypatch):
    _results(monkeypatch, met=False)
    with pytest.raises(ACC.Refused, match=r"'human\|het\|clamp': not met in p1-pooled"):
        ACC.frozen_steps()["human|het|clamp"]
    with pytest.raises(ACC.Refused, match="no such cell"):
        ACC.frozen_steps()["ecoli|hom|clamp"]


def test_p3_refuses_a_step_that_is_not_met_by_name(monkeypatch):
    _results(monkeypatch, met=False)
    with pytest.raises(ACC.Refused, match=r"human\|het\|clamp"):
        ACC.phase_p3_cost(SimpleNamespace(dev=True, naveh_ok=None, naveh_ok_cap=None))
    _results(monkeypatch, met=True)                                                     # the twin reads the step
    st = ACC.p1_steps()["steps"]["human|het|clamp"]
    assert st["dt"] == 3.2e-5 and st["steps_per_individual"] == 2.3e6


def test_p2_refuses_s5s_step_that_is_not_met_by_name(monkeypatch):
    _results(monkeypatch, met=False)
    with pytest.raises(ACC.Refused, match=r"s5\|het\|clamp"):
        ACC.phase_p2_s5_power(SimpleNamespace(dev=True, naveh_ok=None, naveh_ok_cap=None))


def test_the_projection_scales_p1s_rung_cpu_to_the_start_step():
    cells = [{"id": "a", "m_start": 128, "finest": {"m": 64, "kernel_cpu_s": 10.0, "rung_cpu_s": 15.0}},
             {"id": "s5", "m_start": 32, "ladder": [{"m": 16, "draw_cpu_s": 4.0}, {"m": 8, "draw_cpu_s": 2.0}]}]
    pr = PP.projected_cpu_h(cells, k=18)
    assert pr["cells"]["a"]["per_key_s"] == pytest.approx(50.0) and pr["cells"]["s5"]["per_key_s"] == pytest.approx(8.0)
    assert pr["total_cpu_h"] == pytest.approx(18 * 58.0 / 3600)


def test_s5s_short_ladder_extrapolates_its_own_start_step():
    import presets_004 as PR
    with PP.Pool(1) as pool:
        c = PP.s5_ladder(PR.role("s5"), "clamp", pool, rungs=(1, 2), n=300)
    assert c["id"] == "s5|het|clamp" and [r["m"] for r in c["ladder"]] == [1, 2] and c["m_start"] >= 2
    assert 0.5 <= c["p_hat"] <= 1.0 and c["het_spec"] == ("xc", 0.2) and c["base_dt"] == PP.S5_BASE_DT


def test_the_phase_is_registered_with_its_cap():
    assert ACC.PHASES["p1-pooled"] is ACC.phase_p1_pooled and ACC.PILOT_CAPS["p1-pooled"] == 45.0
    assert ("result", "p1-langevin", ("written",)) in ACC.PREREQUISITES["p1-pooled"]


# ------------------------------------------------- p1-pooled-b, the follow-up if p1-pooled caps (T111c, c30e1ca)
IDS = [f"{p}|{h}|{w}" for p in ("mice_F", "Jack_Russell", "ecoli", "human") for h in ("hom", "het")
       for w in ("clamp", "reflect")]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _stub(done):
    p1 = {"status_for_acceptance": "acceptance", "status": "written", "cells": [{"id": i} for i in IDS]}
    prev = {"status_for_acceptance": "acceptance", "status": "capped", "repo_head": "x",
            "cells": {i: {} for i in done}, "frozen": {}}
    return p1, prev


def test_a_capped_p1_pooled_with_two_cells_missing_runs_exactly_those_two(monkeypatch):
    done = [i for i in IDS if i != "human|het|reflect"] + ["s5|het|clamp"]
    p1, prev = _stub(done)
    missing = ACC.p1_pooled_missing(prev, p1)
    assert missing == ["human|het|reflect", "s5|het|reflect"]
    ran, ladders = [], []
    monkeypatch.setattr(PP, "p1_cells", lambda p1_: [{"id": i} for i in IDS])
    monkeypatch.setattr(PP, "with_t_max", lambda c: c)
    monkeypatch.setattr(PP, "s5_ladder", lambda cfg, w, pool: ladders.append(w) or {"id": f"s5|het|{w}",
                                                                                       "worker_cpu_s": 0.0})
    monkeypatch.setattr(PP, "run_cell", lambda c, pool: (ran.append(c["id"]) or {"id": c["id"]}, 0.0))
    b = ACC.stop_rule("p1-pooled-b", 1.0, 15.0, None, clock=Clock())
    recs = ACC.run_p1_pooled_b(b, p1, missing, {}, 1)
    assert ran == ["human|het|reflect", "s5|het|reflect"] and ladders == ["reflect"] and len(recs) == 2


def test_a_complete_p1_pooled_leaves_nothing_for_the_follow_up(monkeypatch):
    p1, prev = _stub(IDS + list(PP.S5_KEYS))
    assert ACC.p1_pooled_missing(prev, p1) == []
    monkeypatch.setattr(ACC, "p1_result", lambda: p1)
    monkeypatch.setattr(ACC, "read_result", lambda phase: prev)
    with pytest.raises(ACC.Refused, match="nothing missing"):
        ACC.phase_p1_pooled_b(SimpleNamespace(dev=True, naveh_ok=None, naveh_ok_cap=None))


def test_the_error_row_is_missing_too():
    p1, prev = _stub([i for i in IDS if i != "ecoli|het|clamp"] + list(PP.S5_KEYS))
    assert ACC.p1_pooled_missing(prev, p1) == ["ecoli|het|clamp"]          # an error row has no finished cell


def _two(monkeypatch, dup=False):
    a = {"status_for_acceptance": "acceptance", "status": "capped",
         "frozen": {"mice_F|hom|clamp": {"m": 170, "dt": 2.5e-4, "met": True, "steps_per_individual": 5e5,
                                         "draw_cpu_s": 300.0, "p_hat": 0.5}}}
    b = {"status_for_acceptance": "acceptance", "status": "written",
         "frozen": {"human|het|reflect": {"m": 36, "dt": 8.7e-5, "met": True, "steps_per_individual": 8e5,
                                          "draw_cpu_s": 600.0, "p_hat": 0.68},
                    **({"mice_F|hom|clamp": a["frozen"]["mice_F|hom|clamp"]} if dup else {})}}
    files = {"p1-pooled": a, "p1-pooled-b": b}
    monkeypatch.setattr(ACC, "committed", lambda rel: any(rel == ACC.result_rel(k) for k in files))
    monkeypatch.setattr(ACC, "read_result", lambda phase: files[phase])


def test_the_accessor_merges_p1_pooled_and_its_follow_up(monkeypatch):
    _two(monkeypatch)
    fs = ACC.frozen_steps()
    assert fs.source == "p1-pooled+p1-pooled-b"
    assert fs["mice_F|hom|clamp"]["m"] == 170 and fs["human|het|reflect"]["m"] == 36
    assert fs["human|het|reflect"]["from"] == "p1-pooled-b"
    with pytest.raises(ACC.Refused, match="no such cell"):
        fs["ecoli|het|clamp"]


def test_the_accessor_refuses_a_cell_in_both(monkeypatch):
    _two(monkeypatch, dup=True)
    with pytest.raises(ACC.Refused, match=r"'mice_F\|hom\|clamp' is in both"):
        ACC.frozen_steps()


def test_the_follow_ups_cap_is_its_projection_plus_25_percent_at_most_15():
    assert ACC.p1_pooled_b_cap(4.0) == pytest.approx(5.0) and ACC.p1_pooled_b_cap(12.0) == 15.0
    assert ACC.p1_pooled_b_cap(30.0) == 15.0 and ACC.PILOT_CAPS["p1-pooled-b"] == 15.0
    assert ACC.P1_POOLED_B["said"] == "2026-11-01: OK a follow-up if it caps (Recommended)"
    cells = [{"id": "a", "m_start": 128, "finest": {"m": 64, "kernel_cpu_s": 10.0, "rung_cpu_s": 15.0}}]
    assert PP.projected_cpu_h(cells, k=18, overhead=1.24)["total_cpu_h"] == pytest.approx(18 * 50.0 * 1.24 / 3600)
    assert ("result", "p1-pooled", ("written", "capped", "incomplete")) in ACC.PREREQUISITES["p1-pooled-b"]
    assert ACC.PHASES["p1-pooled-b"] is ACC.phase_p1_pooled_b
