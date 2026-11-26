"""The mpmath scheme (task T018; research R3, R11, R12): it agrees with the NumPy march where float64 has the
digits, and it exposes the cancellation of an early death's S-difference where float64 does not."""
import json
import os

import mpmath
import numpy as np
import pytest

from reference import mp_scheme as MP
from reference import numpy_scheme as NS

SWEDEN = (3.4531226005888844, 1.8950299204066734, 0.0718812777718253, 0.24659997789670018)
TINY = dict(N=40, dt=0.008, tau_max=2.0)            # stops at step 217
DELTA = 0.02
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "reference", "data", "t7_mpref.json")
STORED = "pure-real-Sweden_F_1910_homo-c0-s0|stored"


@pytest.fixture(scope="module", params=["bdf2", "be"])
def pair(request):
    scheme = request.param
    mp = MP.march(SWEDEN, TINY["N"], TINY["dt"], TINY["tau_max"], scheme=scheme)
    fl = NS.march(TINY["N"], NS.uniform_mesh(TINY["dt"], TINY["tau_max"]), theta=SWEDEN, scheme=scheme)
    return mp, fl


def test_the_march_matches_the_numpy_march_and_conserves_mass(pair):
    mp, fl = pair
    assert mp.k == fl.k and mp.stopped_early and 150 < mp.k < 250
    assert np.array_equal(mp.tau, fl.tau)
    assert max(abs(float(s) - x) for s, x in zip(mp.S_raw, fl.S_raw)) <= 1e-13
    with mpmath.workdps(mp.dps):
        assert max(abs(f + s - 1) for f, s in zip(mp.F_raw, mp.S_raw)) <= mpmath.mpf("1e-35")


def test_the_loglik_equals_the_numpy_one_where_no_term_is_in_round_off(pair):
    """20 ages where 0.05 < S < 0.95 (2/3 deaths with delta = 0.02, 1/3 censored): every term has its digits in
    float64, so all routes of both references agree to 1e-10 nats (measured: 8e-14)."""
    mp, fl = pair
    S, _F = NS.monotone(fl)
    inside = np.flatnonzero((S > 0.05) & (S < 0.95))
    ages = np.linspace(fl.tau[inside[0]], fl.tau[inside[-1]], 20) + 0.0013
    died = np.arange(20) % 3 != 2
    for route in ("auto", "S", "F"):
        ll_mp = MP.loglik(mp, ages, died, DELTA, route)
        ll_np = NS.loglik(fl, ages, died, DELTA, route)
        assert -60 < ll_np < -40
        assert abs(float(ll_mp) - ll_np) <= 1e-10, route


def test_an_early_death_mass_of_1e_14_cancels_in_the_float64_S_difference(pair):
    """At ages 0.05..0.07 the mass of [t, t + 0.02) is 3e-15..4e-14. In mpmath the S- and F-difference routes
    agree (to ~1e-27 relative); float64's F route keeps ~1e-15 relative accuracy, its S route does not (measured
    1e-2 at m = 1.1e-14: a difference of two numbers within 1e-14 of 1)."""
    mp, fl = pair
    ages = np.array([0.05, 0.055, 0.06, 0.065, 0.07])
    m_mp = MP.interval_masses(mp, ages, ages + DELTA)
    m_fl = NS.interval_masses(fl, ages, ages + DELTA)
    rel_S = []
    with mpmath.workdps(mp.dps):
        for j in range(len(ages)):
            mS, mF = m_mp["m_S"][j], m_mp["m_F"][j]
            assert mpmath.mpf("1e-16") < mF < mpmath.mpf("1e-12") and m_mp["use_F"][j] and m_fl["use_F"][j]
            assert abs(mS - mF) <= mpmath.mpf("1e-20") * mF
            assert abs(mpmath.mpf(float(m_fl["m_F"][j])) - mF) <= mpmath.mpf("1e-12") * mF
            rel_S.append(float(abs(mpmath.mpf(float(m_fl["m_S"][j])) - mF) / mF))
    assert min(rel_S) > 1e-6 and max(rel_S) > 1e-3, rel_S


def test_the_cost_estimate_is_printed_before_a_larger_run(capsys):
    est = MP.estimate_seconds(900, 1000)
    assert 0 < est < 3600
    MP.march(SWEDEN, 101, 0.01, 1.0, early_stop=False)          # 101 x 100 node-steps > TINY_NODE_STEPS
    assert "estimated" in capsys.readouterr().out
    MP.march(SWEDEN, 40, 0.01, 0.5)
    assert capsys.readouterr().out == ""


def test_the_copied_reference_loads_with_its_provenance():
    d = json.load(open(DATA))
    p = d["provenance"]
    for key in ("source", "source_md5", "tester_commit", "copied", "method", "theta", "fields", "cohort_and_grid"):
        assert p[key], key
    assert p["source"].endswith("reports/basin/likelihood_fix/v2/t7_mpref.json")
    assert len(p["tester_commit"]) == 40 and len(p["source_md5"]) == 32
    assert set(p["entries_copied"]) == set(d["entries"]) and STORED in d["entries"]
    row, ctx = d["entries"][STORED], p["cohort_and_grid"][STORED]
    assert row["ll_exact"] == pytest.approx(-455596.5253, abs=1e-4)
    assert ctx["grid"] == dict(N=900, dt=0.002, tau_max=2.0, nsteps=1000, Gtol=1e-11, scheme="bdf2", nstart=20)
    assert set(row) >= set(p["fields"])


def test_the_provenance_reproduces_the_stored_value(oracle_v033):
    """The stored rho, grid, L and delta, with the cohort from tester library v1, give ll_exact back through the
    NumPy reference's F route to 1e-8 nats (measured 2.6e-9, which is the 40-digit resolution of ll_exact itself:
    see `remeasured_here`), while its S route is 3,245 nats off."""
    import oracle_v033 as O
    d = json.load(open(DATA))
    row, ctx = d["entries"][STORED], d["provenance"]["cohort_and_grid"][STORED]
    obs = O.observation(STORED.split("|")[0])
    t, died = np.asarray(obs.ages, float), np.asarray(obs.events, int) == 1
    assert t.size == ctx["n_individuals"] and float(obs.delta) == ctx["delta_days"]
    g = ctx["grid"]
    r = NS.march(g["N"], NS.uniform_mesh(g["dt"], g["tau_max"]), theta=tuple(row["rho"]), scheme=g["scheme"],
                 nstart=g["nstart"], Gtol=g["Gtol"])
    assert r.k == row["k_exact"]
    L = ctx["L_days"]
    ll_F = NS.loglik(r, t / L, died, obs.delta / L, route="F")
    ll_S = NS.loglik(r, t / L, died, obs.delta / L, route="S")
    assert abs(ll_F - row["ll_exact"]) <= 1e-8
    assert ll_S - row["ll_exact"] < -1000
    assert abs(ll_F - float(d["remeasured_here"]["mp60_ll_F_route"])) <= 2e-9


# -- per-row terms by definition (review T064, sr-naveh N5) ---------------------------------------------------
@pytest.mark.parametrize("case", ["hazard+entry", "window", "exact", "mixture"])
def test_terms_by_definition_equal_the_engines_v1_terms(case):
    """Where float64 has the digits (masses well above round-off) mpmath's terms by definition and the engine's
    numerics "v1" terms agree to 1e-9 nats: the reference implements the same observation model, so `mpref` can
    judge the low-mass terms of every arm but segments."""
    from numpy.polynomial.hermite_e import hermegauss

    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.engine.inputs import Heterogeneity
    L = 50.0
    rho = (3.4531226005888844, 1.8950299204066734, 0.0718812777718253, 0.24659997789670018)
    p = SRParams(eta=rho[0] / L ** 2, beta=rho[1] / L, epsilon=rho[2] / L, kappa=rho[3], h_ext=0.0)
    ages = np.array([16.0, 22.0, 28.0, 33.0, 37.0, 41.0, 45.0, 52.0])
    died = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    grid = GridPolicy("fixed", N=TINY["N"], dt=TINY["dt"], tau_max=TINY["tau_max"], L=L)
    kw, mk, h = {}, {}, 0.0
    if case == "hazard+entry":
        h = 2e-3
        p = SRParams(**{**p.as_dict(), "h_ext": h})
        coh = Cohort.from_arrays(ages, died, delta=1.0, entry=np.array([0, 5, 10, 30, 0, 20, 44.5, 50]))
        kw = {"fixed": {"h_ext": h}}
        mk = dict(delta=1.0, h=h, entry=np.asarray(coh.entry))
    elif case == "window":
        coh = Cohort.from_arrays(ages, died, delta=1.0, window=(20.0, 45.0))
        mk = dict(delta=1.0, window=(20.0, 45.0))
    elif case == "exact":
        coh = Cohort.from_arrays(ages, died, continuous=True)
        mk = dict(delta=None)
    else:
        coh = Cohort.from_arrays(ages, died, delta=1.0)
        kw = {"heterogeneity": Heterogeneity(parameter="beta", definition="normal_frac_sd", v=0.2, M=3)}
        mk = dict(delta=1.0)
    ev = Evaluator(coh, numerics="v1", grid=grid, **kw)
    got, _ = ev.terms(p)
    if case == "mixture":
        z, w = hermegauss(3)
        marches = [(wk / np.sqrt(2 * np.pi), MP.march((rho[0], rho[1] * (1 + 0.2 * zk), rho[2], rho[3]),
                                                      TINY["N"], TINY["dt"], TINY["tau_max"])) for zk, wk in zip(z, w)]
    else:
        marches = MP.march(rho, TINY["N"], TINY["dt"], TINY["tau_max"])
    ref = MP.terms(marches, ages, died == 1, mk.pop("delta"), L, **mk)
    assert np.max(np.abs(np.asarray(got) - np.array([float(x) for x in ref]))) <= 1e-9
