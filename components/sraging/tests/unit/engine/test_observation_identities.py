"""Observation-model identities (T019; spec SC-004, FR-011 to FR-013; research R8).

Each identity holds by construction of the likelihood, so it is checked exactly where the arithmetic is
the same and to 1e-12 relative where it is rearranged. The "censored record at age a" trick reads
log S_obs(a) from the engine itself: a censored row's term IS log+ S_obs at its age.
"""
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy

pytestmark = pytest.mark.engine
T = np.array([20.0, 35.0, 41.0, 50.0, 52.0, 60.0, 61.0, 70.0, 75.0, 90.0])
E = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
GRID = GridPolicy("fixed", N=120, dt=4e-3, tau_max=3.0, L=50.0)
P = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)
PH = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05, h_ext=2e-3)


def ev(cohort, p=P, **kw):
    return Evaluator(cohort, numerics="v033", grid=GRID, fixed={"h_ext": p.h_ext}, **kw)


def log_S_obs(ages, p, delta=1.0):
    """log+ S_obs at `ages`, read from censored rows (one death keeps the cohort valid); log S(0) = 0."""
    ages = np.asarray(ages, float)
    pos = ages > 0
    a = np.concatenate([ages[pos], [30.0]])
    e = np.concatenate([np.zeros(int(pos.sum()), int), [1]])
    terms, _ = ev(Cohort.from_arrays(a, e, delta=delta), p).terms(p)
    out = np.zeros(ages.size)
    out[pos] = terms[:-1]
    return out


@pytest.mark.parametrize("p", [P, PH], ids=["no hazard", "hazard"])
def test_terms_sum_to_the_total(p):
    e = ev(Cohort.from_arrays(T, E, delta=1.0), p)
    terms, _ = e.terms(p)
    assert e.value(p) == pytest.approx(float(np.sum(terms)), rel=1e-14, abs=0)


@pytest.mark.parametrize("p", [P, PH], ids=["no hazard", "hazard"])
def test_entry_at_zero_is_no_entry(p):
    a = ev(Cohort.from_arrays(T, E, delta=1.0), p).terms(p)[0]
    b = ev(Cohort.from_arrays(T, E, delta=1.0, entry=np.zeros(T.size)), p).terms(p)[0]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("p", [P, PH], ids=["no hazard", "hazard"])
def test_entry_conditions_each_row_on_its_own_survival(p):
    """With a hazard this is S_obs(e) = S(e) e^{-h e}: the hazard before entry cancels (research R8)."""
    en = np.array([0.0, 10.0, 30.0, 40.0, 0.0, 55.0, 5.0, 60.0, 70.0, 80.0])
    a = ev(Cohort.from_arrays(T, E, delta=1.0), p).terms(p)[0]
    b = ev(Cohort.from_arrays(T, E, delta=1.0, entry=en), p).terms(p)[0]
    assert np.allclose(b, a - log_S_obs(en, p), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("p", [P, PH], ids=["no hazard", "hazard"])
def test_a_death_whose_interval_contains_its_entry_is_scored_from_the_entry(p):
    """Death recorded at 50 with delta 1, entered at 50.5: scored over [50.5, 51), conditioned on 50.5."""
    t, e, en = np.array([50.0, 60.0]), np.array([1, 1]), np.array([50.5, 0.0])
    got = ev(Cohort.from_arrays(t, e, delta=1.0, entry=en), p).terms(p)[0][0]
    lo, hi = log_S_obs([50.5, 51.0], p)
    want = math.log(math.exp(lo) - math.exp(hi)) - lo
    assert got == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("p", [P, PH], ids=["no hazard", "hazard"])
def test_window_zero_to_infinity_is_the_full_likelihood(p):
    a = ev(Cohort.from_arrays(T, E, delta=1.0), p).value(p)
    b = ev(Cohort.from_arrays(T, E, delta=1.0, window=(0.0, math.inf)), p).value(p)
    assert a == b


def test_window_excludes_conditions_and_censors_as_srk():
    """SRK's _windowed_ll (numerics v033): rows before t_lo give 0 and are counted; rows after t_hi score
    log S(t_hi); every at-risk row is conditioned on S(t_lo)."""
    c = Cohort.from_arrays(T, E, delta=1.0, window=(30.0, 65.0))
    terms, aux = ev(c, P).terms(P)
    full = ev(Cohort.from_arrays(T, E, delta=1.0), P).terms(P)[0]
    lo, hi = log_S_obs([30.0, 65.0], P)
    assert terms[0] == 0.0 and c.n_excluded_by_window == 1
    assert aux["log_S_lo"] == pytest.approx(lo, rel=1e-13)
    inside = (T >= 30) & (T <= 65)
    assert np.allclose(terms[inside], full[inside] - lo, rtol=1e-13, atol=0)
    beyond = T > 65
    assert np.allclose(terms[beyond], hi - lo, rtol=1e-13, atol=0)


def test_a_chronic_xc_is_the_reduced_parameter_set():
    """Constitution VIII: (xc, eta, beta, eps, kappa) -> (c xc, c eta, c beta, c^2 eps, c kappa)."""
    c, cxc = Cohort.from_arrays(T, E, delta=1.0), 1.7
    reduced = Evaluator(c, numerics="v033", grid=GRID).value(P)
    scaled = SRParams(eta=P.eta * cxc, beta=P.beta * cxc, epsilon=P.epsilon * cxc ** 2,
                      kappa=P.kappa * cxc, xc=cxc)
    got = Evaluator(c, numerics="v033", grid=GRID, fixed={"xc": cxc}).value(scaled)
    assert got == pytest.approx(reduced, rel=1e-12)
