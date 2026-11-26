"""The plants and HR-8 (tasks T074; research R14, R17; gate C3): each plant against SRK's frozen code through the
oracle worker (read-only), beside its twin, at a small n on a synthetic point; the status rule (pass when the plant
shows the defect and the twin does not, fail on the twin, incomplete when not informative); HR-8's bytes and its
B4 plant. NOT APPLICABLE without the snapshot. The phases themselves are never run in the unit suite. W <= 2 here."""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

import plants_004 as PL  # noqa: E402
from sraging import SRParams  # noqa: E402
from sraging import sampler_rules as R  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
BASE = {"t_end": 300.0, "nsteps": 1501}                       # dt 0.2


@pytest.fixture(scope="module")
def srk():
    import oracle_fp_piecewise as OP
    problems = OP.preflight() if os.path.isdir(OP.SNAP) else ["the snapshot is absent"]
    if problems:
        pytest.skip("NOT APPLICABLE: " + "; ".join(problems))
    with OP.session() as s:
        yield PL.CountingSession(s)


def test_the_status_rule():
    assert PL._status(True, True) == "pass" and PL._status(False, True) == "incomplete"
    assert PL._status(True, False) == "fail" == PL._status(False, False)
    assert PL.round_up(np.array([0.2, 0.21, 1.0, np.inf]), 0.5).tolist() == [0.5, 0.5, 1.0, np.inf]


def test_hr9_like_for_like_gives_three_arms_and_hr9s_verdict(srk):
    r = PL.hr9(srk, P, {"dt": 2.0, "m": 16, "t_end": 300.0, "n": 600}, 5)
    assert set(r["verdict"]["statistic"]["arms"]) == {"plant", "twin", "fix"} and r["id"] == "D-402"
    assert all(0 <= r[a]["D"] <= 1 and r[a]["n"] == 600 for a in ("plant", "twin", "fix"))
    assert r["status"] in ("pass", "fail", "incomplete") and srk.cpu_s > 0


def test_b2_srks_realised_step_differs_and_sragings_does_not(srk):
    r = PL.b2(srk, P, {"step_size": 0.07, "t_end": 300.0, "n": 400}, 3)
    assert r["plant"]["realised"] != 0.07 and r["twin"]["equal"] and r["status"] == "pass"


def test_b4_srk_pads_n_mod_n_jobs_and_sraging_keeps_every_one(srk):
    r = PL.b4(srk, P, {"n": 1_003, **BASE}, 3)
    assert r["plant"]["padded"] == 1_003 % r["plant"]["n_jobs"] and r["twin"]["all_present"]
    assert r["status"] == ("pass" if r["plant"]["padded"] else "incomplete") and r["sup_abs_dS"] >= 0


def test_b11_srks_unfiltered_draws_die_at_the_first_step(srk):
    r = PL.b11(srk, P, {"n": 3_000, **BASE}, 3)
    assert r["plant"]["first_step_fraction"] == pytest.approx(PL.B11_PREDICTED, abs=0.012)
    assert r["twin"]["srk_sd_twin"]["first_step_fraction"] < PL.B11_PREDICTED / 10 and r["status"] == "pass"
    assert r["twin"]["sraging_first_step"] < r["plant"]["first_step_fraction"] / 3        # no lump at xc <= 0


def test_seeding_srks_parallel_runs_differ_and_sragings_do_not(srk):
    r = PL.seeding(srk, P, {"n": 400, "W": 2, **BASE}, 3)
    assert r["plant"]["srk_runs_differ"] and r["twin"]["sraging_identical"] and r["status"] == "pass"


def test_the_hazard_plant_predicts_its_size_first(srk):
    r = PL.hazard(srk, P, {"n": 500, "nsteps": 1501, "dt": 0.2}, 3)
    assert r["plant"]["h"] == pytest.approx(0.05 / 0.2) and r["twin"]["h"] == pytest.approx(1e-4 / 0.2)
    assert r["plant"]["predicted_sup_dS"] > r["twin"]["predicted_sup_dS"] >= 0
    assert r["status"] in ("pass", "fail", "incomplete")


def test_recording_srk_on_the_outer_grid_sraging_on_delta(srk):
    r = PL.recording(srk, P, {"n": 500, "dt": 1.0, "m": 2, "delta": 0.25, "nsteps": 301}, 3)
    assert r["plant"]["srk_on_Dt_grid"] == 1.0 and r["plant"]["sraging_off_Dt_grid"] > 0 and r["status"] == "pass"


def test_d409_getsrhetros_default_is_heterogeneous(srk):
    r = PL.d409(srk, P, {"n": 3_000, **BASE}, 3)
    assert r["plant"]["default_xc_var"] == 0.2 and r["twin"]["sraging_default_homogeneous"]
    assert r["status"] == "pass"


def test_d410_the_clamp_leaves_zeros_at_a_young_age_and_reflect_none():
    r = PL.d410(P, {"n": 1_000, "dt": 0.5, "age": 4.0, "t_max": 300.0}, 3)
    assert r["plant"]["p0"] > 0 and r["twin"]["p0"] == 0.0 and r["status"] == "pass"


def test_d411_getisrs_recorder_writes_zero_for_the_dead(srk):
    r = PL.d411(srk, P, {"n": 800, "dt": 0.5, "age": 100.0}, 3)
    assert r["plant"]["zeros_among_dead"] > 0 and not r["plant"]["holds"] and r["twin"]["holds"]
    assert r["status"] == "pass" and 0 <= r["sup_abs_dS"] <= 1


def test_hr8_bytes_and_its_plant():
    rows, plants = PL.hr8_rows(P, {"n": 1_003, "dt": 0.5, "W": [1, 2]}, 3)
    assert [r["W"] for r in rows] == [1, 2] and all(r["equal"] for r in rows)
    assert plants == [{"W": 2, "dropped": 1, "differs": True}]
    assert R.hr8_verdict(rows, plants).status == "pass"


def test_the_phases_are_registered():
    import acceptance_004 as ACC
    assert ACC.PHASES["plants"] is ACC.phase_plants and ACC.PHASES["hr8"] is ACC.phase_hr8
    assert math.isclose(PL.B11_PREDICTED, 0.02275, rel_tol=1e-3)


def test_hr9s_fix_arm_runs_at_the_frozen_step(srk):
    """Amendment A4 (Naveh 2026-11-03): the fix arm at sraging's frozen step, recorded through the same Delta t grid."""
    r = PL.hr9(srk, P, {"dt": 2.0, "m": 16, "t_end": 300.0, "n": 400, "fix_dt": 0.05}, 5)
    assert r["fix"]["dt"] == 0.05
    twin = PL.hr9(srk, P, {"dt": 2.0, "m": 16, "t_end": 300.0, "n": 400}, 5)       # the twin: Delta t / 16
    assert twin["fix"]["dt"] == 2.0 / 16


def test_each_plant_is_priced_at_its_own_steps():
    rates = {"lifetime": {"mice_F": 124.8}, "ns_sraging": 1.18e-8, "ns_srk": 3.15e-8}
    spec = {"hr9": {"preset": "mice_F", "dt": 2.3566, "m": 16, "n": 25_000, "fix_dt": 2.518e-4},
            "b2": {"preset": "mice_F", "step_size": 0.0428, "n": 25_000},
            **{k: {"preset": "mice_F", "n": 25_000, "t_end": 214, "nsteps": 5000} for k in ("b4", "b11", "seeding",
                                                                                           "d409")},
            "hazard": {"preset": "mice_F", "n": 25_000, "nsteps": 5000, "dt": 0.0428},
            "recording": {"preset": "mice_F", "n": 25_000, "dt": 1.0, "delta": 0.25, "nsteps": 215},
            "d410": {"preset": "mice_F", "n": 25_000, "age": 0.428, "dt": 0.0428},
            "d411": {"preset": "mice_F", "n": 25_000, "dt": 0.0428, "age": 109.2}}
    h = PL.projected_cpu_h(spec, rates)
    fix = 25_000 * 124.8 / 2.518e-4 * 1.18e-8 / 3600
    assert fix < h < 0.2                       # the fix arm at the frozen step dominates; far under the formula's 3.2
    assert PL.plant_steps("hr9", spec["hr9"]) == ([2.3566 / 16] * 2, [2.3566 / 16, 2.518e-4])
