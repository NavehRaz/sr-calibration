"""Curves (T057; spec FR-009, US6): S and f against SRK's frozen fpt_full_density on the same grid; the
"model" policy's L against SRK's auto_L; curves with features equal the evaluator's own quantities."""
import math

import numpy as np
import pytest

import oracle_fp_piecewise as OP
from sraging import Cohort, SRParams
from sraging.engine.curves import curves, model_L
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import AgeSegments, Heterogeneity, Segment

pytestmark = [pytest.mark.parity, pytest.mark.engine]
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)


@pytest.fixture(scope="module")
def srk():
    try:
        with OP.session() as s:
            yield s
    except OP.NotApplicable as ex:
        pytest.skip(f"NOT APPLICABLE: {'; '.join(ex.problems)}")


@pytest.mark.parametrize("scheme", ["bdf2", "be"])
def test_S_and_f_on_the_grid_match_fpt_full_density(srk, scheme):
    L = 124.0
    ev = Evaluator(Cohort.from_arrays([50.0], [1], continuous=True), numerics="v033",
                   grid=GridPolicy("fixed", N=900, dt=1e-3, tau_max=20.0, L=L), scheme=scheme)
    tau, S, f, k = ev.survival_on_grid(P)
    r = srk.call("fpt_full_density", rho_eta=L ** 2 * P.eta, rho_beta=L * P.beta, rho_eps=L * P.epsilon,
                 kappa=P.kappa, N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11, scheme=scheme)
    n = np.asarray(r["S"]).size
    assert n == k + 1
    assert np.max(np.abs(S[:n] - np.asarray(r["S"]))) <= 1e-8
    assert np.max(np.abs(f[:n] - np.asarray(r["f"]))) <= 1e-6 * np.max(np.asarray(r["f"]))


def test_model_L_matches_srk_auto_L(srk):
    """SRK fp_curves.auto_L (probe N=300, dt=4e-3, BE): the same rule with the engine's march."""
    ours = model_L(P, scheme="be")
    assert math.isfinite(ours) and ours > 0
    # the defining property: at L = the model's mean, SRK's own solver gives a dimensionless mean of ~1
    r = srk.call("fpt_full", rho_eta=ours ** 2 * P.eta, rho_beta=ours * P.beta,
                 rho_eps=ours * P.epsilon, kappa=P.kappa, N=900, dt=1e-3, tau_max=20.0, Gtol=1e-11, scheme="be")
    assert abs(float(r["mean"]) - 1.0) < 0.02         # at L = model mean, the dimensionless mean is ~1


def test_curves_with_features_equal_the_evaluators_own_quantities():
    ages = np.array([0.0, 20.0, 60.0, 100.0, 140.0, 200.0])
    p = SRParams(**{**P.as_dict(), "h_ext": 1e-3})
    segs = AgeSegments(Segment(id="b", parameter="beta", value=0.5, window=(60.0, 120.0)))
    het = Heterogeneity(parameter="eta", definition="normal_frac_sd", v=0.2, M=5)
    cv = curves(p, ages, segments=segs, heterogeneity=het)
    assert cv.grid.policy == "model" and cv.S[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(cv.S) <= 0) and np.all(cv.f >= 0)
    ok = cv.S > 0
    assert np.allclose(cv.hazard[ok], cv.f[ok] / cv.S[ok])
    plain = curves(P, ages, grid=GridPolicy("fixed", N=900, dt=1e-3, tau_max=20.0, L=124.0))
    ev = Evaluator(Cohort.from_arrays(ages[1:], np.ones(5, int), continuous=True), numerics="v1",
                   grid=GridPolicy("fixed", N=900, dt=1e-3, tau_max=20.0, L=124.0))
    _, aux = ev.terms(P)
    assert np.allclose(plain.S[1:], aux["S_i"], rtol=0, atol=0) and np.allclose(plain.f[1:], aux["mass"], rtol=0, atol=0)
