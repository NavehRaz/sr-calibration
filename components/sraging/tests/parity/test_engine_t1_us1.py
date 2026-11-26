"""T1 for User Story 1 (T021; spec SC-002; Clarifications): numerics "v033" against SRK's NumPy likelihood,
like for like (same grid, same scheme), through the frozen `fp_piecewise_cb191f5` oracle in its worker.

Per evaluation: total |dloglik| <= 1e-5 nats; each resolved term <= 1e-8 nats; max |dS| <= 1e-8 on the
march grid. "Resolved" is derived, not chosen: the two solvers agree in S to dS_max (measured here, per
evaluation, from fpt_full), so a term of mass m can differ by about dS_max/m; terms with
m >= 4*dS_max/1e-8 are held to 1e-8, and the rest to that bound (their mpmath check is the `mpref` phase).
Fast subset: 2 cohorts per arm (pure, hazard, window with on-grid edges, censor) x 2 theta x 2 schemes.
The full T1 panel is tests/acceptance/panel_002_t1.toml (tools/acceptance_002.py t1).
"""
import math

import numpy as np
import pytest

import oracle_fp_piecewise as OP
import oracle_v033 as O
from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy

pytestmark = [pytest.mark.parity, pytest.mark.engine]
TOTAL, TERM, DS = 1e-5, 1e-8, 1e-8
# (cohort_id, ExtH in age units or None): the first medium-tier cohort per arm at kappa 0.01 and at another
# kappa (library order; window edges on the delta grid), from the tester's reports/validity/validity.csv.
CASES = [("pure-k0.01-p00-c0-s0", None), ("pure-k2-p04-c0-s0", None),
         ("hazard-k0.01-p00-c0-s0", 5.5), ("hazard-k0.0316-p00-c0-s0", 5.5),
         ("window-k0.01-p00-c0-s0", None), ("window-k0.0316-p00-c0-s0", None),
         ("censor-k0.01-p00-c0-s0", None), ("censor-k0.0316-p00-c0-s0", None)]


@pytest.fixture(scope="module")
def srk():
    try:
        with OP.session() as s:
            yield s
    except OP.NotApplicable as ex:
        pytest.skip(f"NOT APPLICABLE: {'; '.join(ex.problems)}")


def _setup(cid, ext, theta_label):
    obs = O.observation(cid)
    t, e, d = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int), obs.delta
    tr = dict(O.library().truth(cid))
    if theta_label == "eta x1.5":
        tr["eta"] *= 1.5
    h = 0.0 if ext is None else math.exp(-ext)
    g0 = GridPolicy("v033").grid(Cohort.from_arrays(t, e, delta=d))
    return t, e, d, obs.window, tr, h, ext, g0


@pytest.mark.parametrize("scheme", ["bdf2", "be"])
@pytest.mark.parametrize("theta_label", ["truth", "eta x1.5"])
@pytest.mark.parametrize("cid,ext", CASES, ids=[c for c, _ in CASES])
def test_t1_against_srk(oracle_v033, srk, cid, ext, theta_label, scheme):
    t, e, d, win, tr, h, ext, g0 = _setup(cid, ext, theta_label)
    grid = GridPolicy("fixed", N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, L=g0.L)
    ev = Evaluator(Cohort.from_arrays(t, e, delta=d, window=win), numerics="v033", grid=grid, scheme=scheme,
                   fixed={"h_ext": h})
    p = SRParams(**tr, h_ext=h)
    kw = dict(N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, scheme=scheme)
    ref = srk.call("fp_loglik_physical", death_times=t, events=e, eta=tr["eta"], beta=tr["beta"],
                   epsilon=tr["epsilon"], xc=1.0, kappa=tr["kappa"], L=g0.L,
                   external_hazard=math.inf if ext is None else ext,
                   time_range=list(win) if win else None, sampling_dt=d, **kw)
    assert abs(ev.value(p) - float(ref["value"])) <= TOTAL

    # S on the march grid (the SR survival, before the hazard)
    tau, S, _, _ = ev.survival_on_grid(p)
    L = g0.L
    fp = srk.call("fpt_full", rho_eta=L ** 2 * tr["eta"], rho_beta=L * tr["beta"], rho_eps=L * tr["epsilon"],
                  kappa=tr["kappa"], **kw)
    S_ref = np.asarray(fp["S"])
    dS = float(np.max(np.abs(S[:S_ref.size] - S_ref)))
    assert dS <= DS

    if win is None:                         # per term against SRK-P at zero intervention (= fp_loglik)
        terms, aux = ev.terms(p)
        rt = srk.call("piecewise_loglik_terms", death_times=t, events=e,
                      theta=dict(eta=tr["eta"], beta=tr["beta"], epsilon=tr["epsilon"], kappa=tr["kappa"], xc=1.0),
                      interventions=[], L=L, t_end=float(t.max()),
                      external_hazard=math.inf if ext is None else ext, sampling_dt=d, **kw)
        diff = np.abs(terms - np.asarray(rt["terms"]))
        mass = np.where(e == 1, aux["mass"], aux["S_i"])
        cut = 4.0 * max(dS, 1e-16) / TERM
        resolved = mass >= cut
        assert np.all(diff[resolved] <= TERM), f"worst resolved term {diff[resolved].max():.3e}"
        bound = 4.0 * max(dS, 1e-16) / np.maximum(mass, 1e-300)
        assert np.all(diff[~resolved] <= np.maximum(bound[~resolved], TERM))
