"""HR-10's phase `paths` (tasks T098; research R24, R25; contracts/acceptance.md §4): 8 configurations (each preset
homogeneous and at xc sd 0.2) x 2 ages near S = 0.75 and 0.4 on the dt grid, both walls at P1's steps with m_max = 2m,
judged on the dt/2 rung; the eta(1 + delta) plant sized for a predicted sup|dF| = 2 x 2.02 / sqrt(n_alive) and its
delta = 0 twin; the deterministic checks; SRK's zero-for-dead recorder through the NaN check. The configurations are
frozen at freeze-L ([paths]); a spec of that shape is built here. W <= 2 here (the coordinator's load rule)."""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

import paths_004 as PA  # noqa: E402
import s4_fp_langevin_004 as S4  # noqa: E402
from sraging import SRParams  # noqa: E402
from sraging import sampler_rules as R  # noqa: E402

PRESETS = ("mice_F", "Jack_Russell", "ecoli", "human")
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, h_ext=2e-3)
SPEC = {"presets": list(PRESETS), "ages": {p: [40.0, 100.0] for p in PRESETS}, "n": 25_000, "G": 16, "W": [1, 7, 15],
        "key": 4100, "plant": {"preset": "mice_F", "het": "hom", "age": 100.0, "delta": 0.05}, "srk_n": 2_000}


def p1(missing=(), dt=0.05):
    steps = {S4.step_key(p, h, w): {"m": 8, "dt": dt} for p in PRESETS for h in ("hom", "het")
             for w in ("clamp", "reflect")}
    for k in missing:
        steps.pop(k)
    return {"steps": steps, "cells": [{"preset": p, "het": h} for p in PRESETS for h in ("hom", "het")]}


def test_the_configurations_are_8_per_wall_at_p1s_steps():
    rows = PA.build_configs(SPEC, p1())
    for wall in ("clamp", "reflect"):
        w = [r for r in rows if r["wall"] == wall]
        assert len(w) == 8 and len({r["id"] for r in w}) == 8
        assert sorted((r["preset"], str(r["parameter"]), r["sd"]) for r in w) == sorted(
            (p, "None", 0.0) if h == "hom" else (p, "xc", 0.2) for p in PRESETS for h in ("hom", "het"))
    assert all(r["m"] == 8 and r["dt"] == 0.05 and r["m_max"] == 16 and r["n"] == 25_000 for r in rows)
    assert all(r["ages"] == [40.0, 100.0] for r in rows)
    assert sum(len(r["ages"]) for r in rows if r["wall"] == "clamp") == SPEC["G"]


def test_a_configuration_without_a_p1_step_or_with_ages_off_its_grid_is_refused():
    with pytest.raises(PA.Refused, match="no P1 step"):
        PA.build_configs(SPEC, p1(missing=(S4.step_key("ecoli", "het", "reflect"),)))
    with pytest.raises(PA.Refused, match="off the dt"):
        PA.build_configs({**SPEC, "ages": {**SPEC["ages"], "human": [40.0, 100.01]}}, p1())
    with pytest.raises(PA.Refused, match="G"):
        PA.build_configs({**SPEC, "G": 12}, p1())


def test_the_ages_are_near_s_075_and_04_on_the_dt_grid():
    dt = 0.25
    ages = PA.pick_ages(P, None, dt)
    assert len(ages) == 2 and ages[0] < ages[1]
    assert all(abs(round(a / dt) * dt - a) < 1e-12 for a in ages)
    S = PA.observed_survival(P, None, np.array(ages))
    assert abs(S[0] - 0.75) < 0.01 and abs(S[1] - 0.4) < 0.01


def test_the_plants_delta_is_sized_for_twice_the_detection_shift():
    d = PA.plant_delta(P, 100.0, 25_000)
    n_alive = 25_000 * d["S_obs"]
    assert d["target"] == pytest.approx(2 * 2.02 / math.sqrt(n_alive))
    assert d["predicted_sup_dF"] == pytest.approx(d["target"], rel=0.02) and 0 < d["delta"] < 0.5


@pytest.fixture(scope="module")
def cfg_result():
    row = {"id": "t/hom/clamp", "preset": "t", "parameter": None, "sd": 0.0, "wall": "clamp", "n": 3_000,
           "ages": [20.0, 100.0], "m": 1, "dt": 0.5, "m_max": 2, "p1_step": "t|hom|clamp"}
    return row, PA.run_config(row, P, 7, 0.5)


def test_a_configuration_gives_hr10_test_rows_judged_on_the_dt2_rung(cfg_result):
    row, res = cfg_result
    assert res["wall"] == "clamp" and 1 <= res["m"] <= row["m_max"] and len(res["ladder"]) in (1, 2)
    assert [t["age"] for t in res["tests"]] == row["ages"]
    for t in res["tests"]:
        assert set(t) >= {"id", "n_alive", "p0", "u_c", "x_c", "D_c", "D", "D_bias", "p_hat", "m", "m_max",
                          "reference_grid_error"}
        assert t["D_c"] <= t["D"] + 1e-15 and t["n_alive"] > 0 and t["p_hat"] == 0.5
    assert res["nan_check"]["holds"]
    v = R.hr10_verdict(res["tests"], {"nan_iff_dead": res["nan_check"], "times_equal": {"holds": True},
                                      "bytes_by_W": {"holds": True}}, [], wall="clamp", cpu_spent=0.0)
    assert v.status in ("pass", "fail", "incomplete") and v.wall == "clamp"


def test_the_plant_and_its_twin_are_judged_by_the_walls_statistic():
    row = {"id": "t/hom", "preset": "t", "parameter": None, "sd": 0.0, "wall": "reflect", "n": 2_000, "m": 1,
           "dt": 0.5, "m_max": 2}
    pl = PA.run_plant(row, P, 100.0, 0.3, 11, kind="plant")
    tw = PA.run_plant(row, P, 100.0, 0.0, 12, kind="twin")
    assert pl["kind"] == "plant" and tw["kind"] == "twin" and pl["wall"] == "reflect"
    assert pl["statistic"] == pl["D"] and pl["n_alive"] > 0 and tw["delta"] == 0.0
    assert pl["statistic"] > tw["statistic"]                     # a 30 % eta shift is visible at n_alive ~ 1,000


def test_the_deterministic_checks_hold_for_sraging():
    row = {"id": "t/hom/clamp", "wall": "clamp", "n": 701, "ages": [20.0, 100.0], "dt": 0.5}
    c = PA.deterministic_checks(row, P, 5, W_list=[1, 2], chunk=200)
    assert c["times_equal"]["holds"] and c["bytes_by_W"]["holds"] and c["bytes_by_W"]["by_W"] == {"1": True, "2": True}


class FakeSRK:
    """SRK's getISR recorder as the oracle op returns it: zeros for the dead (D-411)."""

    def __init__(self, zeros_for_dead=True):
        self.zeros, self.calls = zeros_for_dead, []

    def call(self, op, **kw):
        self.calls.append((op, kw))
        n, ages = kw["npeople"], np.asarray(kw["ages"], float)
        t = np.linspace(1.0, 1.5 * ages.max(), n)
        X = np.where(t[:, None] <= ages[None, :], 0.0 if self.zeros else np.nan, 0.2)
        return {"death_times": t, "events": np.ones(n, int), "trajectories": X, "ages": ages}


def test_srks_recorder_row_fails_the_nan_check():
    srk = FakeSRK()
    r = PA.srk_recorder(srk, P, [40.0, 100.0], n=300, dt=0.5, seed=3)
    assert r["kind"] == "srk_recorder" and not r["nan_check"]["holds"] and r["nan_check"]["zeros_among_dead"] > 0
    op, kw = srk.calls[0]
    assert op == "simulate_isr_paths" and kw["external_hazard"] == pytest.approx(-math.log(P.h_ext))
    assert PA.srk_recorder(FakeSRK(zeros_for_dead=False), P, [40.0], n=50, dt=0.5, seed=3)["nan_check"]["holds"]


def test_the_phase_is_registered():
    import acceptance_004 as ACC
    assert ACC.PHASES["paths"] is ACC.phase_paths


def test_srks_recorder_extra_runs_on_the_plant_configurations_clamp_row(monkeypatch):
    """At 252ad3a the srk extra (no wall) looked up a row for wall None and raised StopIteration, so the phase read
    incomplete with every test passing (T125); behind it, [paths.ages] was indexed by preset alone."""
    import contextlib
    import acceptance_004 as ACC
    import oracle_fp_piecewise as OP
    srk = FakeSRK()
    monkeypatch.setattr(OP, "session", lambda: contextlib.nullcontext(srk))
    spec = {"plant": {"preset": "mice_F", "het": "hom", "age": 100.0}, "srk_n": 300, "key": 5, "W": [1, 7],
            "ages": {"mice_F|hom": [40.0, 100.0], "mice_F|het": [30.0, 90.0]}}
    rows = [{"wall": "clamp", "dt": 0.25, "preset": "human", "sd": 0.0, "ages": [70.0]},
            {"wall": "clamp", "dt": 0.5, "preset": "mice_F", "sd": 0.2, "ages": [30.0, 90.0]},
            {"wall": "reflect", "dt": 0.125, "preset": "mice_F", "sd": 0.0, "ages": [40.0, 100.0]},
            {"wall": "clamp", "dt": 1.0, "preset": "mice_F", "sd": 0.0, "ages": [40.0, 100.0]}]
    r, cpu = ACC.paths_extra(("srk", 0, None), rows, spec, {}, P, None)
    assert r["kind"] == "srk_recorder" and r["dt"] == 1.0 and r["n"] == 300 and cpu == 0.0
    kw = srk.calls[0][1]
    assert kw["seed"] == 5 + 20_000 and kw["ages"] == [40.0, 100.0] and not r["nan_check"]["holds"]
    with pytest.raises(StopIteration):                                  # twin: a missing plant row still refuses
        ACC.paths_extra(("srk", 0, None), rows[:3], spec, {}, P, None)


# ------------------------------------------- Naveh 2026-11-03 (ff794f5): E. coli out of the clamp tests; amendment A2
def test_ecoli_has_no_clamp_configuration_and_ages_are_per_preset_and_class():
    spec = {**SPEC, "clamp_excluded": ["ecoli"],
            "ages": {**{f"{p}|hom": [40.0, 100.0] for p in PRESETS}, **{f"{p}|het": [35.0, 95.0] for p in PRESETS}}}
    rows = PA.build_configs(spec, p1())
    assert not any(r["preset"] == "ecoli" and r["wall"] == "clamp" for r in rows)
    assert sum(r["preset"] == "ecoli" and r["wall"] == "reflect" for r in rows) == 2                # reflect stays
    by = {(r["preset"], r["sd"] > 0, r["wall"]): r for r in rows}
    assert by[("mice_F", False, "clamp")]["ages"] == [40.0, 100.0] and by[("mice_F", True, "clamp")]["ages"] == [35.0, 95.0]
    assert len([r for r in PA.build_configs(SPEC, p1()) if r["wall"] == "clamp"]) == 8            # the twin


def test_the_named_clamp_tests_are_judged_on_reflect_only():
    spec = {"reflect_only": [{"cell": "Jack_Russell|het|clamp", "ages": [0]}, {"cell": "ecoli|hom|clamp", "ages": [0, 1]},
                             {"cell": "ecoli|het|clamp", "ages": [0, 1]}]}
    assert PA.reflect_only(spec) == {("Jack_Russell|het|clamp", 0), ("ecoli|hom|clamp", 0), ("ecoli|hom|clamp", 1),
                                     ("ecoli|het|clamp", 0), ("ecoli|het|clamp", 1)}
    row = {"id": "t/v0/sd0/clamp", "preset": "t", "parameter": None, "sd": 0.0, "wall": "clamp", "n": 1_200,
           "ages": [20.0, 100.0], "m": 1, "dt": 0.5, "m_max": 2, "p1_step": "t|hom|clamp"}
    frozen = {"p_hat": 1.0, "pooled_b": 0.0008, "D_key_mean": 0.0015}
    r = PA.run_frozen(row, P, 7, frozen, skip_ages=[0])
    assert [t["age"] for t in r["tests"]] == [100.0] and [t["age"] for t in r["not_judged_on_this_wall"]] == [20.0]
    assert all(t["frozen_b"] == 0.0008 and "path_bias" in t for t in r["tests"])
    v = R.hr10_verdict(r["tests"], {"nan_iff_dead": r["nan_check"], "times_equal": {"holds": True},
                                    "bytes_by_W": {"holds": True}}, [], wall="clamp", cpu_spent=0.0)
    assert v.n_valid + v.n_unassessable == 1                              # the skipped age is not a clamp test
    twin = PA.run_frozen(row, P, 7, frozen)                               # the twin: both ages judged
    assert len(twin["tests"]) == 2 and twin["not_judged_on_this_wall"] == []
