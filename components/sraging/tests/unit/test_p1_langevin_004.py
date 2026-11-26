"""P1's tool (tasks T072; research R19; gate C1, C4): the 16 cells at the presets' base steps, a rung's measurements
(the coupled D, KS, CPU, the lump p0 and the paths' step bias at the HR-10 ages), the weak order taken in [1/2, 1],
the step at crit/10, the clamp's extension, the noise floor, Dd's overhead, B1's lag, SRK's re-measure through a fake
session, and the result's keys (what acceptance_004.p1_result documents). Small n here; P1 itself is never run in
the unit suite."""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

import p1_langevin_004 as P1  # noqa: E402
import presets_004 as PR  # noqa: E402
from sraging import SRParams  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)


def test_the_grid_is_4_presets_x_hom_het_x_both_walls_at_their_base_steps():
    cs = P1.cells()
    assert len(cs) == 16 and len({c["id"] for c in cs}) == 16 and P1.M_RUNGS == (1, 2, 4, 8, 16)
    assert {(c["preset"], c["het"], c["wall"]) for c in cs} == {(p, h, w) for p in PR.NAMES for h in ("hom", "het")
                                                                for w in ("clamp", "reflect")}
    assert all(c["base_dt"] == PR.base_dt(c["preset"]) and c["n"] == 25_000 for c in cs)
    assert all((c["parameter"], c["sd"]) == ((None, 0.0) if c["het"] == "hom" else ("xc", 0.2)) for c in cs)


@pytest.mark.parametrize("p_true,want", [(0.3, 0.5), (0.75, 0.75), (1.6, 1.0)])
def test_p_hat_is_the_log_log_slope_taken_in_a_half_to_one(p_true, want):
    rungs = [{"dt": 0.5 / m, "D": 0.04 * (0.5 / m) ** p_true} for m in (1, 2, 4, 8, 16)]
    f = P1.fit_p_hat(rungs)
    assert f["p_raw"] == pytest.approx(p_true) and f["p_hat"] == pytest.approx(want)
    assert P1.fit_p_hat(rungs[:1])["p_hat"] == 0.5                         # one point: the conservative 1/2


def test_the_step_is_the_largest_dt_whose_judged_bias_meets_crit_over_10():
    cr = 1.973 / math.sqrt(25_000)
    rungs = [{"m": m, "dt": 0.5 / m, "D": 3 * (cr / 10) / m} for m in (1, 2, 4, 8, 16)]        # p = 1: b = D
    st = P1.choose_step(rungs, 1.0, "clamp")
    assert st["met"] and st["m"] == 4 and st["judged_bias"] <= cr / 10 and st["crit"] == pytest.approx(cr)
    miss = P1.choose_step([{**r, "D": 100 * r["D"]} for r in rungs], 1.0, "clamp")
    assert not miss["met"] and miss["m"] is None and miss["finest_m"] == 16


@pytest.fixture(scope="module")
def small_cell():
    cell = {"id": "t|hom|clamp", "preset": "t", "het": "hom", "parameter": None, "sd": 0.0, "wall": "clamp",
            "base_dt": 0.5, "n": 1_500}
    return P1.run_cell(cell, P, 21, rungs=(1, 2), extend=(4,))


def test_a_cell_records_its_rungs_p_step_and_the_paths_at_the_hr10_ages(small_cell):
    c = small_cell
    assert [r["m"] for r in c["ladder"]] == [1, 2] and len(c["ages"]) == 2
    assert all(abs(a / 0.5 - round(a / 0.5)) < 1e-12 for a in c["ages"])
    for r in c["ladder"]:
        assert r["dt"] == 0.5 / r["m"] and r["D"] >= 0 and r["ks_D_half"] is not None and r["draw_cpu_s"] > 0
        assert r["ns_per_individual_step"] > 0 and r["steps_per_individual"] > 0
        assert [x["age"] for x in r["paths"]] == c["ages"] and all(0 <= x["p0"] <= 1 for x in r["paths"])
        assert all(x["path_D"] is not None for x in r["paths"])
    assert 0.5 <= c["fit"]["p_hat"] <= 1.0
    assert c["step"]["met"] or [r["m"] for r in c["extension"]] == [4]          # the clamp's extension (gate C4)


def test_the_result_carries_what_the_langevin_phases_read(small_cell, monkeypatch):
    monkeypatch.setitem(PR.PRESETS, "t", {**PR.PRESETS["mice_F"],
                                          "params": {k: v for k, v in P.as_dict().items() if k != "time_unit"}})
    out = P1.result([small_cell])
    k = small_cell["id"]
    assert set(out) >= {"steps", "cells", "draw_cpu_s", "ns_per_individual_step", "steps_per_individual",
                        "p_hat_cells", "p_hat", "host", "cell_ages"}
    assert out["cell_ages"][k] == small_cell["ages"]                   # ROLES["plants"]["d411"]'s age reads it
    assert out["steps"][k]["met"] == small_cell["step"]["met"] and out["draw_cpu_s"][k] > 0
    assert {c["m"] for c in out["p_hat_cells"]} == {r["m"] for r in small_cell["ladder"] + small_cell["extension"]}
    assert all(c["p_hat"] == small_cell["fit"]["p_hat"] and c["config"]["wall"] == "clamp" for c in out["p_hat_cells"])


def test_the_noise_floor_dd_and_b1s_lag():
    cell = {"id": "t|hom|clamp", "preset": "t", "het": "hom", "parameter": None, "sd": 0.0, "wall": "clamp",
            "base_dt": 0.5, "n": 800}
    nf = P1.noise_floor(cell, P, [1, 2, 3], m=1)
    assert len(nf["D"]) == 3 and nf["floor"] == pytest.approx(np.mean(nf["D"])) and nf["crit_over_30"] > 0
    dd = P1.dd_overhead(P, 0.5, n=800)
    assert 0 < dd["overhead_fraction"] < 1 and dd["per_individual_ns"] > 0
    b1 = P1.b1_lag(P, None, 0.5)
    assert b1["f_max"] > 0 and b1["lag_at_base_dt"] == pytest.approx(b1["f_max"] * 0.25)
    assert b1["dt_for_target"] == pytest.approx(0.04 / b1["f_max"])


class FakeSRK:
    def __init__(self):
        self.calls = []

    def call(self, op, **kw):
        self.calls.append((op, kw))
        if op == "simulate_sr_hetro":
            return {"death_times": np.linspace(1.0, 100.0, kw["npeople"]), "events": np.ones(kw["npeople"], int),
                    "cpu_s": 7.0}
        return {"value": -1.0, "cpu_s": 0.4}


def test_srks_re_measure_is_n_15000_one_core_at_m_9_and_xc_sd_02():
    srk = FakeSRK()
    r = P1.srk_remeasure(srk, "Jack_Russell")
    (op1, a), (op2, b) = srk.calls
    assert op1 == "simulate_sr_hetro" and a["npeople"] == 15_000 and a["xc_var"] == 0.2
    assert a["steps"] == 5000 and a["time_step_multiplier"] == 2 and a["dt"] == 24 / 4999
    assert a["external_hazard"] == 5.577 and a["kappa"] == PR.PRESETS["Jack_Russell"]["params"]["kappa"]
    assert op2 == "fp_loglik_het_physical" and b["M"] == 9 and b["het_var"] == 0.2 and b["het_param"] == "xc"
    assert r["langevin_cpu_s"] == 7.0 and r["fp_mixture_cpu_s"] == 0.4 and r["n_jobs"] == 1
    srk2 = FakeSRK()
    P1.srk_remeasure(srk2, "mice_F")                                       # no ExtH: SRK's kernel kills none
    assert "external_hazard" not in srk2.calls[0][1] and srk2.calls[0][1]["time_step_multiplier"] == 1


def test_the_phase_is_registered():
    import acceptance_004 as ACC
    assert ACC.PHASES["p1-langevin"] is ACC.phase_p1_langevin and ACC.PILOT_CAPS["p1-langevin"] == 6.0


def test_srks_re_measure_runs_against_the_snapshot_at_a_small_n():
    import oracle_fp_piecewise as OP
    problems = OP.preflight() if os.path.isdir(OP.SNAP) else ["the snapshot is absent"]
    if problems:
        pytest.skip("NOT APPLICABLE: " + "; ".join(problems))
    with OP.session() as s:
        r = P1.srk_remeasure(s, "mice_F", n=300)
    assert r["langevin_cpu_s"] > 0 and r["fp_mixture_cpu_s"] > 0 and math.isfinite(r["fp_value"])
