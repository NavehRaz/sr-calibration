"""profile_v033's own drop-one leverage check (spec 006 FR-024, US5; research R17; tasks T039): the terms, G and the
Hessian from the engine at numerics "v033" on v0.3.3's grid -- which is v0.3.3's objective bit for bit -- 003f's rule
unchanged, the refits v0.3.3's MLE stage (the replay) and its own warm start. Fit-free pieces run fast; the ITP
example's age-0 death, 003f's positive control, is flagged by a real v033 fit (slow)."""
import os

import numpy as np
import pytest

from sraging import Cohort, fit_cohort
from sraging import leverage as LEV
from sraging.leverage_rule import RULE

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ITP = os.path.join(REPO, "tests", "data", "itp_plus_age0.csv")
Z = np.log([3.6e-5, 0.032, 0.0024, 0.001])


@pytest.fixture(scope="module")
def itp():
    return Cohort.from_csv(ITP, delta=1.0, time_unit="days")


def test_the_engines_v033_terms_are_v033s_objective(itp):
    td = LEV.terms_and_derivatives_v033(itp, Z)
    obj, _box = LEV.v033_objective(itp)
    assert float(np.sum(td["terms"])) == pytest.approx(-float(obj.nll(Z)), rel=1e-13, abs=0)
    assert td["G"].shape == (itp.n, 4) and np.allclose(td["H"], td["H"].T) and td["terms"].size == itp.n
    assert np.array_equal(td["inputs"]["t"], np.asarray(itp.ages, float))


def test_a_dropped_row_leaves_v033s_objective_on_the_other_rows(itp):
    obj, _ = LEV.v033_objective(itp, 0)
    full, _ = LEV.v033_objective(itp)
    assert len(obj.t) == itp.n - 1 if hasattr(obj, "t") else True
    ll = LEV.reduced_loglik_v033(itp, 0, [Z])[0]
    assert ll != -float(full.nll(Z)) and np.isfinite(ll)


def test_003fs_rule_is_the_one_applied():
    assert RULE["quantities"] == "all4" and RULE["ratio"] == 1.5 and RULE["delta_total_nats"] == 1.92


def test_no_theta_hat_is_not_run_never_a_pass(itp):
    out = LEV.check_v033(itp, None, {})
    assert out["label"] == "not_run" and out["reason"]


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="measured 2026-11-10 (review/evidence/v033_itp_control.json): v033's theta-hat "
                   "on this cohort is not stationary (free gradient 0.175 per individual, g0 = 0), the top-influence "
                   "row is the oldest death, R = 1.29 < 1.5, and 003g's rule reads not_assessed; R-9's positive "
                   "control goes to Naveh at the freeze (tasks T058)")
def test_v033_flags_the_itp_age0_death():
    """R-9's positive control: as 003f's check does for J, v033's flags the age-0 death (row 1)."""
    r = fit_cohort(Cohort.from_csv(ITP, delta=1.0, time_unit="days"), calibrator="profile_v033")
    lev = r.diagnostics["leverage"]
    assert lev["label"] == "flag" and lev["dropped"]["row"] == 1 and lev["dropped"]["age"] == 0.0
    assert "LEVERAGE" in [w.code for w in r.warnings]
    assert lev["provenance"]["calibrator"] == "profile_v033" and lev["cost"]["cpu_s"] > 0


def test_v033s_flag_message_carries_its_failed_validation():
    """Naveh 2026-11-15 (sr-naveh S7): profile_v033's check failed R-9, so its flag says so; J's does not. The caveat
    closes the flag's message only, and profile_v033 passes it."""
    import ast

    from sraging.warnings import V033_LEVERAGE_CAVEAT, leverage_warnings
    flag = {"label": "flag", "dropped": {"row": 1, "age": 0.0, "event": 1}, "ratios": {}, "delta_total": 2.0}
    (w,) = leverage_warnings(flag, caveat=V033_LEVERAGE_CAVEAT)
    assert w.message.endswith(f"({V033_LEVERAGE_CAVEAT})") and "R-9" in w.message
    assert "R-9" not in leverage_warnings(flag)[0].message
    assert "R-9" not in leverage_warnings({"label": "failed", "reason": "x"}, caveat=V033_LEVERAGE_CAVEAT)[0].message
    src = open(os.path.join(os.path.dirname(LEV.__file__), "calibrators", "profile_v033.py")).read()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "leverage_warnings"]
    assert calls and all(any(k.arg == "caveat" and getattr(k.value, "id", None) == "V033_LEVERAGE_CAVEAT"
                             for k in c.keywords) for c in calls)
