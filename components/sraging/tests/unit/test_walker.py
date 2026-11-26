"""The combination walker against a brute-force constrained maximisation (spec 003, T026; SC-004; FR-015, FR-017).

Slow (opt-in, `pytest -m slow`): an MLE and one combination walk on the small cohort, then the brute-force profile
around each walker endpoint (16 Sobol starts plus the walker's own starts per point). The pass rule is SC-004's: no
endpoint narrower than brute force by more than 0.01 dex, none wider by more than 0.05 dex. A planted walker whose
inner step never optimises must fail it (constitution V). The full comparison (64 starts, 20 cohorts) is the
`walker` acceptance phase.
"""
import math
import os

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import qmc

from sraging import Cohort
from sraging.claims import combinations as K
from sraging.claims import walker as W

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")
DEX = math.log(10.0)


@pytest.fixture(scope="module")
def problem():
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import estimator as est, fp_jax as fj
    from sraging.compat.v033.fp_curvefit import fp_curvefit_seed
    c = Cohort.from_csv(CSV, delta=1.0)
    inp = PJ.fit_inputs(c, "profile_jax2")
    obj = est._Objective(fj, inp["t"], inp["e"], inp["L"], inp["kw"], inp["box"], hazard=False, window=inp["window"])
    box4 = {p: inp["box"][p] for p in ("eta", "beta", "epsilon", "kappa")}
    bounds = [(math.log(box4[p][0]), math.log(box4[p][1])) for p in ("eta", "beta", "epsilon", "kappa")]
    s = fp_curvefit_seed(inp["t"], inp["e"], xc=1.0)
    seed = np.log(np.clip([s["eta"], s["beta"], s["epsilon"], s["k"]], 1e-12, None))
    r = minimize(lambda z: obj.nll_grad(z), seed, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 400, "ftol": 1e-13, "gtol": 1e-6})
    return obj, r.x, -float(r.fun), bounds, box4, seed


def brute_profile(obj, key, c, bounds, extra, n=16, seed=0):
    sob = qmc.Sobol(4, scramble=True, seed=seed).random(n)
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    starts = [lo + u * (hi - lo) for u in sob] + list(extra)
    r = W._best(obj, key, c, starts, bounds)
    return -np.inf if r is None else r[1]


def brute_endpoint(obj, key, c_end, ll_hat, bounds, extra, outward):
    """The crossing of 2 (ll_hat - profile) = 3.8415 near c_end, from brute-force profile values 0.005 dex apart."""
    cs = [c_end + outward * k * 0.005 * DEX for k in range(-4, 5)]
    devs = [2.0 * (ll_hat - brute_profile(obj, key, c, bounds, extra, seed=i)) for i, c in enumerate(cs)]
    thr = 3.8415
    for (c0, d0), (c1, d1) in zip(zip(cs, devs), zip(cs[1:], devs[1:])):
        if (d0 - thr) * (d1 - thr) <= 0 and d0 != d1:
            s0, s1 = math.sqrt(max(d0, 0)), math.sqrt(max(d1, 0))
            return c0 + (math.sqrt(thr) - s0) / (s1 - s0) * (c1 - c0)
    return None


@pytest.mark.slow
def test_the_walk_matches_brute_force_within_sc004(problem):
    obj, z_hat, ll_hat, bounds, box4, seed = problem
    key = "eta_over_eps"
    w = W.walk(obj, key, z_hat, ll_hat, bounds=bounds, box4=box4, seed_z=seed)
    ci = w["ci"]
    extra = [np.asarray(z) for _, _, z in w["nodes"]]
    checked = 0
    for end, outward in ((ci["ci_lo"], -1), (ci["ci_hi"], +1)):
        if not math.isfinite(end):
            continue
        bf = brute_endpoint(obj, key, math.log(end), ll_hat, bounds, extra, outward)
        assert bf is not None
        narrower_dex = outward * (bf - math.log(end)) / DEX          # > 0: the walker stops inside brute force
        assert narrower_dex <= 0.01 and -narrower_dex <= 0.05, (end, math.exp(bf), narrower_dex)
        checked += 1
    assert checked >= 1


@pytest.mark.slow
def test_a_walker_whose_inner_step_never_optimises_fails(problem, monkeypatch):
    obj, z_hat, ll_hat, bounds, box4, seed = problem
    key = "eta_over_eps"

    def lazy(obj, key, c, z0, bounds):          # planted: project and stop, no constrained maximisation
        z = K.project(key, z0, c)
        nll, _ = obj.nll_grad(z)
        return (z, -float(nll)) if np.isfinite(nll) else None
    monkeypatch.setattr(W, "_solve", lazy)
    w = W.walk(obj, key, z_hat, ll_hat, bounds=bounds, box4=box4, seed_z=seed)
    ci = w["ci"]
    monkeypatch.undo()
    bad = []
    for end, outward in ((ci["ci_lo"], -1), (ci["ci_hi"], +1)):
        if not math.isfinite(end):
            bad.append("no endpoint")
            continue
        bf = brute_endpoint(obj, key, math.log(end), ll_hat, bounds, [z_hat], outward)
        if bf is None or abs(bf - math.log(end)) / DEX > 0.01:
            bad.append(end)
    assert bad


@pytest.mark.slow
def test_a_point_above_the_mle_is_reported_as_a_gain(problem):
    """FR-017: the walk reports how far its best node beats the theta-hat it was given (the planted 3-nat shortfall)."""
    obj, z_hat, ll_hat, bounds, box4, seed = problem
    w = W.walk(obj, "eta_over_eps", z_hat, ll_hat - 3.0, bounds=bounds, box4=box4, seed_z=seed)
    assert w["gain"] > 1.92
