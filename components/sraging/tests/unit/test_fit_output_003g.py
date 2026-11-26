"""What a default (jax2 1.0.1) fit says about what it cannot see (spec 003g FR-011 to FR-013; contracts/fit-output.md;
tasks T017-T019). Each new label, line or note fires on its planted case and not on its twin."""
import copy
import io
import json
import os

from sraging import leverage as LEV
from sraging import warnings as W
from sraging.calibrators import profile_jax2 as PJ
from sraging.leverage_rule import G_CRIT_PER_IND
from sraging.result import FitResult

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")


# ----------------------------------------------------------------------------------- FR-011: the not_assessed label
def g0_block(g0):
    return {"status": "ok", "g0": g0, "below_start": False}


#: the 10 fits measured (|g|, n): 003b's pilot (lev_pilot_003g.json) and spec 003g's fits 1 and 2 (their results)
ASSESSED = {"ITP+age0": (0.001783, 305), "fit1 censor-k0.01-p05-c1-s1": (0.018702, 10_000)}
NOT_ASSESSED = {"Sweden_F_1910_hetro-c1-s1": (16.97, 25_000), "combined_human_F-c1-s1": (26.4, 25_000),
                "Sweden_F_1910_hetro-c1-s0": (641.8, 25_000), "combined_human_F-c1-s0": (155.9, 25_000),
                "Denmark_F_1900_hetro-c1-s1": (10.1, 25_000), "Sweden_M_1910_hetro-c0-s0": (267.1, 100_000),
                "Sweden_F_1910_hetro-c0-s0": (474.5, 100_000), "fit2 realfull-Sweden_F_1910_homo": (646.77, 23_448)}


def test_g_crit_per_individual_separates_the_measured_cases():
    """DV-3 (Naveh 2026-10-28): |g| is a sum over individuals and grows with n, so the criterion is per individual.
    Assessed 1.9e-6 to 5.9e-6, not assessed 4.0e-4 to 2.6e-2 [M]: 5e-5 is about a decade from each group."""
    assert max(g / n for g, n in ASSESSED.values()) < G_CRIT_PER_IND < min(g / n for g, n in NOT_ASSESSED.values())


def test_no_step_at_a_non_stationary_theta_hat_is_not_assessed_and_its_twin_is_not():
    for g, n in NOT_ASSESSED.values():
        assert LEV.not_assessed("no_flag", g0_block(0.0), g, n)                 # planted: the human cohorts
    for g, n in ASSESSED.values():
        assert not LEV.not_assessed("no_flag", g0_block(0.0), g, n)            # twins: ITP+age0, fit 1
    assert LEV.not_assessed("flat_valley_slide", g0_block(0.0), 642.0, 25_000)
    assert not LEV.not_assessed("no_flag", g0_block(0.4), 10.0, 25_000)        # the warm start moved: assessed
    assert not LEV.not_assessed("flag", g0_block(0.0), 10.0, 25_000)           # a flag is never replaced
    assert not LEV.not_assessed("failed", g0_block(0.0), 10.0, 25_000)
    assert not LEV.not_assessed("no_flag", {"status": "failed"}, 10.0, 25_000)


def test_the_same_pathology_at_small_n_is_still_not_assessed():
    """The reason for DV-3: Denmark's |g| = 10.1 at n = 25,000 scales to ~0.40 at n = 1,000, which the registered
    |g| > 1.0 read as no_flag; per individual it is the same 4e-4."""
    g = 10.1 * 1_000 / 25_000
    assert g < 1.0 and LEV.not_assessed("no_flag", g0_block(0.0), g, 1_000)


def test_not_assessed_warns_and_no_flag_does_not():
    block = {"label": "not_assessed", "reason": "r",
             "at_theta_hat": {"free_gradient_norm": 12.3, "free_gradient_per_individual": 4.92e-4},
             "warm_termination": {"status": "ok", "scipy_status": 2, "nit": 0, "message": "ABNORMAL: "}}
    (w,) = W.leverage_warnings(block)
    assert w.code == "LEVERAGE" and "could not assess this fit" in w.message
    assert "the optimiser could not move from theta-hat (gradient 0.000492 per individual)" in w.message
    assert "not stationary" not in w.message
    assert "a single influential death would not be seen" in w.message
    assert W.leverage_warnings(dict(block, label="no_flag")) == ()


# --------------------------------------------------------------------------------------- FR-013: the DT_RULE line
def _example(**dt):
    d = json.load(open(EXAMPLE))
    d["diagnostics"]["dt_floor"]["grid_rule"].update(dt)
    return FitResult.from_json(json.dumps(d))


def _printed(r):
    from sraging import cli
    out = io.StringIO()
    cli._print_jax2(r, out)
    return out.getvalue()


def test_the_dt_rule_line_prints_in_scope_and_its_twin_prints_nothing():
    inside = _printed(_example(dt_in_scope=True, cv_c=0.0123))
    assert "DT_RULE: in scope (CV_c 0.0123 < 0.05); dt capped at CV/80" in inside
    assert "DT_RULE" not in _printed(_example(dt_in_scope=False, cv_c=0.18))



def test_a_sharp_101_fit_prints_the_measured_limits_on_the_dt_rule_line_and_its_twins_do_not():
    """N2 (Naveh 2026-10-29), planted: in DT_RULE's regime the note carries the measured limits and prints on the DT_RULE
    line (once, not again as a plain note). Twins: outside the regime, on _h, and on 1.0.0 there is no note."""
    from sraging.warnings import SHARP_COHORT_LIMIT

    class J:
        HAZARD = False

    class H:
        HAZARD = True
    inside = {"dt_floor": {"grid_rule": {"dt_in_scope": True, "cv_c": 0.0181}}}
    note = PJ.sharp_limit_note("1.0.1", J, inside)
    assert note == SHARP_COHORT_LIMIT and note is not SHARP_COHORT_LIMIT
    assert PJ.sharp_limit_note("1.0.1", J, {"dt_floor": {"grid_rule": {"dt_in_scope": False}}}) is None
    assert PJ.sharp_limit_note("1.0.1", H, inside) is None
    assert PJ.sharp_limit_note("1.0.0", J, inside) is None
    for want in ("132 of 133 pure and 43 of 45 censored", "no upper end on 49 of 227", "raw beta was closed on 6 of 227",
                 "read r-bar/eta, which is not truth-scored on 1.0.1", "Below CV_c 0.01 only 2 cohorts"):
        assert want in note["message"]
    r = _example(dt_in_scope=True, cv_c=0.0181)
    r.diagnostics["notes"] = list(r.diagnostics.get("notes") or []) + [note]
    printed = _printed(r)
    dt_line = printed.index("DT_RULE: in scope")
    assert printed.count("measured on 227 benchmark cohorts") == 1
    assert printed.index("  limits in this regime: measured on 227") > dt_line

# ------------------------------------------------------------------------------ FR-013: the _h stratum sentence
def test_the_h_note_says_in_the_stratum_for_a_sharp_sr_cv_and_not_in_for_its_twin():
    inside = PJ.h_stratum_note(0.03)
    outside = PJ.h_stratum_note(0.08)
    assert inside["code"] == "HAZARD_SHARP_STRATUM" and "is in the sharp-hazard stratum" in inside["message"]
    assert "SR CV 0.03 < 0.05" in inside["message"]
    assert "is not in the sharp-hazard stratum" in outside["message"] and "SR CV 0.08 >= 0.05" in outside["message"]
    assert PJ.h_stratum_note(None)["data"]["sr_cv"] is None and "could not be computed" in PJ.h_stratum_note(None)["message"]


def test_the_sr_cv_at_a_known_point():
    """The converged SR CV (sraging-gen-1's grid) at the README example's theta-hat: 0.180 [M, 2026-10-22]."""
    d = json.load(open(EXAMPLE))
    from sraging.params import SRParams
    th = {k: d["theta_hat"][k] for k in ("eta", "beta", "epsilon", "kappa")}
    assert abs(PJ.sr_cv(SRParams(**th, h_ext=0.0, time_unit="days")) - 0.1803) < 5e-4


# ---------------------------------------------------------------------------- FR-012: the combination-claim limit
def test_a_101_fit_with_combination_claims_carries_the_limit_and_its_twins_do_not():
    note = PJ.combination_limit_note("1.0.1", {"on": True})
    assert note["code"] == "COMBINATIONS_NOT_TRUTH_SCORED" and "not truth-scored on 1.0.1" in note["message"]
    assert "0.818" in note["message"]
    assert PJ.combination_limit_note("1.0.1", {"on": False}) is None           # twin: no combination claims
    assert PJ.combination_limit_note("1.0.0", {"on": True}) is None            # twin: 1.0.0 was scored (spec 003)


def test_the_no_combinations_note_names_the_scored_version():
    from sraging import NO_COMBINATIONS_NOTE
    assert "truth-scored" in NO_COMBINATIONS_NOTE and "1.0.0" in NO_COMBINATIONS_NOTE
    assert "1.0.1" in NO_COMBINATIONS_NOTE
